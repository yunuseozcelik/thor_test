#!/usr/bin/env python3
"""
Ollama LLM Benchmark Tool
Tam otomatik benchmark aracı — Windows, Linux, Jetson AGX Thor destekli.
Kullanım: python ollama_benchmark.py [--skip-download] [--models "m1,m2"] [--quick] [--output prefix]
"""

import argparse
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("requests kütüphanesi gerekli: pip install requests")
    sys.exit(1)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

# ── Renk yardımcıları ──────────────────────────────────────────────────────

def _c(text, color):
    if not HAS_COLOR:
        return str(text)
    return f"{color}{text}{Style.RESET_ALL}"

def green(t):  return _c(t, Fore.GREEN)
def red(t):    return _c(t, Fore.RED)
def yellow(t): return _c(t, Fore.YELLOW)
def cyan(t):   return _c(t, Fore.CYAN)
def bold(t):   return _c(t, Fore.WHITE + Style.BRIGHT) if HAS_COLOR else str(t)

# ── Sabitler ────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"

DEFAULT_MODELS = [
    "gemma3:12b",
    "gemma3:27b",
    "qwen3:32b",
    "deepseek-r1:70b",
]

BENCHMARKS = {
    "A": {
        "name": "Basit Soru (Kısa Cevap)",
        "prompt": "What is 2+2? Answer in one word.",
    },
    "B": {
        "name": "Askeri Senaryo Analizi",
        "prompt": (
            "You are a military AI system installed in an armored vehicle. "
            "Analyze this situation: 3 soldiers moving north at 200m distance, "
            "1 armored vehicle approaching from east at 500m, 2 unknown objects "
            "detected on thermal camera at 150m south. Provide threat assessment "
            "and recommended action. Be concise, max 5 sentences."
        ),
    },
    "C": {
        "name": "Uzun Metin Üretimi",
        "prompt": (
            "Write a detailed technical specification for an AI-powered smart "
            "armored vehicle surveillance system. Include: camera requirements, "
            "processing unit specs, object detection capabilities, communication "
            "systems, and power requirements. Write at least 300 words."
        ),
    },
    "D": {
        "name": "Türkçe Anlama",
        "prompt": (
            "Sen bir askeri yapay zeka sistemisin. Zırhlı aracın termal kamerasında "
            "3 hareket eden nesne tespit edildi. Kuzey yönünde 200 metre mesafede. "
            "Tehdit değerlendirmesi yap ve önerilen aksiyonu Türkçe olarak bildir. "
            "Kısa ve net ol."
        ),
    },
    "E": {
        "name": "Kod Üretimi",
        "prompt": (
            "Write a Python function that takes a list of detected objects with "
            "their coordinates (x, y, distance, type) and returns the nearest "
            "threat. Include type hints and a brief docstring."
        ),
    },
}

# ── Global durum ────────────────────────────────────────────────────────────

results = {}          # model -> test -> metrikler
interrupted = False
gpu_info_cache = None  # bir kere topla, tekrar kullan


def signal_handler(_sig, _frame):
    global interrupted
    print(yellow("\n\n⚠ Ctrl+C algılandı — mevcut sonuçlar kaydediliyor..."))
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)

# ── Yardımcı: komut çalıştır ───────────────────────────────────────────────

def run_cmd(cmd, timeout=30):
    """Komutu çalıştır, (returncode, stdout, stderr) döndür."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return -1, "", "command not found"
    except subprocess.TimeoutExpired:
        return -2, "", "timeout"
    except Exception as e:
        return -3, "", str(e)

# ── Platform tespiti ────────────────────────────────────────────────────────

def is_jetson():
    """Jetson platformunda mıyız?"""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
        return "jetson" in model or "nvidia" in model
    except (FileNotFoundError, PermissionError):
        pass
    # tegrastats varlığına bak
    rc, _, _ = run_cmd("which tegrastats")
    return rc == 0


def detect_jetson_model():
    """Jetson kart modelini döndür."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            return f.read().strip().rstrip("\x00")
    except Exception:
        return "NVIDIA Jetson (unknown model)"

# ── Sistem bilgisi ──────────────────────────────────────────────────────────

def get_system_info():
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "N/A",
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "is_jetson": is_jetson(),
    }

    # CPU bilgisi
    if platform.system() == "Linux":
        rc, out, _ = run_cmd("nproc")
        if rc == 0:
            info["cpu_cores"] = int(out)
        rc, out, _ = run_cmd("lscpu")
        if rc == 0:
            for line in out.splitlines():
                if "Model name" in line:
                    info["cpu_name"] = line.split(":", 1)[1].strip()
                    break
    elif platform.system() == "Windows":
        info["cpu_name"] = platform.processor()
        info["cpu_cores"] = os.cpu_count() or 0
    else:
        info["cpu_cores"] = os.cpu_count() or 0

    # RAM
    try:
        if platform.system() == "Linux":
            rc, out, _ = run_cmd("free -b")
            if rc == 0:
                for line in out.splitlines():
                    if line.startswith("Mem:"):
                        total = int(line.split()[1])
                        info["ram_gb"] = round(total / (1024**3), 1)
                        break
        elif platform.system() == "Windows":
            rc, out, _ = run_cmd(
                'wmic computersystem get TotalPhysicalMemory /value'
            )
            if rc == 0:
                m = re.search(r"TotalPhysicalMemory=(\d+)", out)
                if m:
                    info["ram_gb"] = round(int(m.group(1)) / (1024**3), 1)
    except Exception:
        pass

    # Jetson bilgisi
    if info["is_jetson"]:
        info["jetson_model"] = detect_jetson_model()

    return info


def get_gpu_info():
    """GPU bilgisini al. Jetson veya masaüstü NVIDIA destekler."""
    global gpu_info_cache
    if gpu_info_cache is not None:
        return gpu_info_cache

    info = {"available": False, "name": "N/A", "vram_mb": 0, "driver": "N/A", "type": "none"}

    # ── Jetson ──
    if is_jetson():
        info["available"] = True
        info["type"] = "jetson"
        info["name"] = detect_jetson_model()
        info["driver"] = "Jetson integrated"
        # Jetson'da VRAM = sistem RAM'inin tamamı (unified memory)
        try:
            rc, out, _ = run_cmd("free -b")
            if rc == 0:
                for line in out.splitlines():
                    if line.startswith("Mem:"):
                        total = int(line.split()[1])
                        info["vram_mb"] = int(total / (1024**2))
                        break
        except Exception:
            pass
        # jtop varsa daha iyi bilgi
        try:
            rc, out, _ = run_cmd("jetson_release")
            if rc == 0:
                info["jetson_release"] = out[:500]
        except Exception:
            pass
        gpu_info_cache = info
        return info

    # ── Masaüstü NVIDIA ──
    rc, out, _ = run_cmd("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if rc == 0 and out:
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) >= 3:
            info["available"] = True
            info["type"] = "nvidia"
            info["name"] = parts[0]
            try:
                info["vram_mb"] = int(parts[1])
            except ValueError:
                info["vram_mb"] = 0
            info["driver"] = parts[2]

    gpu_info_cache = info
    return info


def sample_gpu_usage():
    """Anlık GPU kullanım ve VRAM oku."""
    usage = {"gpu_util": -1, "vram_used_mb": -1, "vram_total_mb": -1}

    if is_jetson():
        # tegrastats tek satır al
        try:
            rc, out, _ = run_cmd("tegrastats --interval 100 --count 1", timeout=5)
            if rc != 0:
                # Bazı Jetson'larda tegrastats root gerektirir
                rc, out, _ = run_cmd("sudo tegrastats --interval 100 --count 1", timeout=5)
            if rc == 0 and out:
                # RAM kullanımı: "RAM 5678/15832MB"
                ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", out)
                if ram_match:
                    usage["vram_used_mb"] = int(ram_match.group(1))
                    usage["vram_total_mb"] = int(ram_match.group(2))
                # GR3D (GPU) kullanımı: "GR3D_FREQ 45%"
                gr3d = re.search(r"GR3D_FREQ\s+(\d+)%", out)
                if gr3d:
                    usage["gpu_util"] = int(gr3d.group(1))
        except Exception:
            pass
        return usage

    # Masaüstü NVIDIA
    rc, out, _ = run_cmd(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits"
    )
    if rc == 0 and out:
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        if len(parts) >= 3:
            try:
                usage["gpu_util"] = int(parts[0])
                usage["vram_used_mb"] = int(parts[1])
                usage["vram_total_mb"] = int(parts[2])
            except ValueError:
                pass
    return usage

# ── Ollama kontrol ─────────────────────────────────────────────────────────

def check_ollama():
    """Ollama kurulu ve çalışıyor mu kontrol et."""
    rc, out, _ = run_cmd("ollama --version")
    if rc != 0:
        print(red("✗ Ollama kurulu değil!"))
        print("  Şuradan indirin: https://ollama.com")
        print("  Jetson için:     https://ollama.com/download/linux")
        sys.exit(1)
    version = out.strip()
    print(green(f"✓ Ollama bulundu: {version}"))

    # Serve çalışıyor mu?
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            print(green("✓ Ollama sunucusu çalışıyor"))
            return True
    except requests.ConnectionError:
        pass

    print(yellow("⚠ Ollama sunucusu yanıt vermiyor, başlatılıyor..."))
    # Arka planda başlat
    if platform.system() == "Windows":
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    # Bağlantı bekle
    for i in range(15):
        time.sleep(1)
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
            if r.status_code == 200:
                print(green("✓ Ollama sunucusu başlatıldı"))
                return True
        except requests.ConnectionError:
            pass
        print(f"  Bekleniyor... ({i+1}/15)")

    print(red("✗ Ollama sunucusu başlatılamadı. Lütfen ayrı bir terminalde 'ollama serve' çalıştırın."))
    sys.exit(1)


def get_local_models():
    """Yerel modellerin listesini döndür."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

# ── Model indirme ──────────────────────────────────────────────────────────

def pull_model(model_name):
    """Modeli indir, progress göster."""
    print(cyan(f"\n  ↓ {model_name} indiriliyor..."))
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/pull",
            json={"name": model_name},
            stream=True, timeout=600,
        )
        last_status = ""
        for line in r.iter_lines():
            if interrupted:
                return False
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = data.get("status", "")
            if "completed" in data and "total" in data:
                total = data["total"]
                completed = data["completed"]
                if total > 0:
                    pct = completed / total * 100
                    bar_len = 30
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    size_str = f"{completed/(1024**3):.1f}/{total/(1024**3):.1f} GB"
                    print(f"\r    [{bar}] {pct:5.1f}% {size_str}  ", end="", flush=True)
            elif status != last_status:
                print(f"\r    {status:<60}", end="", flush=True)
                last_status = status
        print()
        print(green(f"  ✓ {model_name} hazır"))
        return True
    except requests.exceptions.Timeout:
        print(red(f"\n  ✗ {model_name} indirme zaman aşımı"))
        return False
    except Exception as e:
        print(red(f"\n  ✗ {model_name} indirilemedi: {e}"))
        return False


def ensure_models(models, skip_download):
    """Gerekli modellerin hazır olduğundan emin ol."""
    local = get_local_models()
    # Normalize: "gemma3:4b" eşleşmesi — tag dahil kontrol
    local_set = set()
    for m in local:
        local_set.add(m)
        # "gemma3:4b" ile "gemma3:4b" direkt eşleşir
        # "gemma3:4b-latest" gibi varyantlar da kontrol edilebilir

    ready = []
    gpu = get_gpu_info()
    vram = gpu.get("vram_mb", 0)
    # Jetson'da unified memory — 12b model çalışabilir ama yine de uyaralım
    is_jetson_device = gpu.get("type") == "jetson"

    for model in models:
        if interrupted:
            break

        # VRAM kontrolü — 12b modeli için
        if "12b" in model:
            if not is_jetson_device and 0 < vram < 8192:
                print(yellow(f"  ⊘ {model} atlanıyor — VRAM yetersiz ({vram} MB < 8192 MB)"))
                continue
            if is_jetson_device and 0 < vram < 16384:
                print(yellow(f"  ⊘ {model} atlanıyor — Unified memory yetersiz ({vram} MB)"))
                continue

        # Zaten var mı?
        found = any(model == m or model == m.split(":")[0] for m in local_set)
        if not found:
            # "gemma3:1b" -> local'de "gemma3:1b" olarak arayalım
            found = model in local_set

        if found:
            print(green(f"  ✓ {model} zaten mevcut"))
            ready.append(model)
            continue

        if skip_download:
            print(yellow(f"  ⊘ {model} mevcut değil (--skip-download)"))
            continue

        if pull_model(model):
            ready.append(model)

    return ready

# ── GPU izleme thread'i ────────────────────────────────────────────────────

class GpuMonitor:
    """Benchmark sırasında arka planda GPU istatistiklerini toplar."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            s = sample_gpu_usage()
            if s["gpu_util"] >= 0:
                self.samples.append(s)
            self._stop.wait(0.5)

    def summary(self):
        if not self.samples:
            return {"avg_gpu_util": -1, "max_vram_mb": -1, "avg_vram_mb": -1}
        utils = [s["gpu_util"] for s in self.samples if s["gpu_util"] >= 0]
        vrams = [s["vram_used_mb"] for s in self.samples if s["vram_used_mb"] >= 0]
        return {
            "avg_gpu_util": round(sum(utils) / len(utils), 1) if utils else -1,
            "max_vram_mb": max(vrams) if vrams else -1,
            "avg_vram_mb": round(sum(vrams) / len(vrams)) if vrams else -1,
        }

# ── Benchmark çalıştırma ───────────────────────────────────────────────────

def run_benchmark(model, test_id, prompt):
    """Tek bir benchmark testini Ollama API ile çalıştır."""
    gpu_mon = GpuMonitor()
    gpu_mon.start()

    t_start = time.perf_counter()
    ttft = None  # time to first token
    full_response = ""
    token_count = 0
    eval_count = 0
    eval_duration_ns = 0
    prompt_eval_duration_ns = 0

    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": 1024},
            },
            stream=True, timeout=300,
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if interrupted:
                break
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk = data.get("response", "")
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                full_response += chunk
                print(chunk, end="", flush=True)

            # Son mesajda istatistikler gelir
            if data.get("done", False):
                eval_count = data.get("eval_count", 0)
                eval_duration_ns = data.get("eval_duration", 0)
                prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)

    except requests.exceptions.Timeout:
        print(red("\n  [TIMEOUT]"))
    except Exception as e:
        print(red(f"\n  [HATA: {e}]"))

    t_end = time.perf_counter()
    gpu_mon.stop()
    total_time = t_end - t_start

    # Token tahmini
    word_count = len(full_response.split())
    estimated_tokens = int(word_count * 1.3) if eval_count == 0 else eval_count
    tps = estimated_tokens / total_time if total_time > 0 else 0

    # Ollama'dan gelen eval ile gerçek token/s
    if eval_count > 0 and eval_duration_ns > 0:
        real_tps = eval_count / (eval_duration_ns / 1e9)
    else:
        real_tps = tps

    gpu_summary = gpu_mon.summary()

    return {
        "model": model,
        "test_id": test_id,
        "test_name": BENCHMARKS[test_id]["name"],
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 3) if ttft else None,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
        "eval_count": eval_count,
        "tokens_per_sec": round(real_tps, 1),
        "prompt_eval_ms": round(prompt_eval_duration_ns / 1e6, 1) if prompt_eval_duration_ns else 0,
        "avg_gpu_util": gpu_summary["avg_gpu_util"],
        "max_vram_mb": gpu_summary["max_vram_mb"],
        "avg_vram_mb": gpu_summary["avg_vram_mb"],
        "response_preview": full_response[:200],
        "response_length": len(full_response),
    }

# ── Warmup ──────────────────────────────────────────────────────────────────

def warmup_model(model):
    """Modeli belleğe yükle — ilk çağrıda model load süresi ölçülür."""
    print(cyan(f"  ⏳ {model} yükleniyor (warmup)..."), end="", flush=True)
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}},
            timeout=120,
        )
        t1 = time.perf_counter()
        load_time = round(t1 - t0, 2)
        print(green(f" {load_time}s"))
        return load_time
    except Exception as e:
        print(red(f" Hata: {e}"))
        return -1

# ── Rapor ───────────────────────────────────────────────────────────────────

def print_header(text):
    w = 70
    print("\n" + bold("=" * w))
    print(bold(f"  {text}"))
    print(bold("=" * w))


def print_system_report(sys_info, gpu):
    print_header("SİSTEM BİLGİLERİ")
    if sys_info.get("is_jetson"):
        print(cyan(f"  Platform  : {sys_info.get('jetson_model', 'Jetson')}"))
    else:
        print(f"  Platform  : {sys_info['platform']} {sys_info['platform_release']}")
    print(f"  CPU       : {sys_info.get('cpu_name', sys_info.get('processor', 'N/A'))}")
    print(f"  CPU Cores : {sys_info.get('cpu_cores', 'N/A')}")
    print(f"  RAM       : {sys_info.get('ram_gb', 'N/A')} GB")
    print(f"  Python    : {sys_info['python_version']}")
    print(f"  Arch      : {sys_info['architecture']}")
    print()
    if gpu["available"]:
        if gpu["type"] == "jetson":
            print(green(f"  GPU       : {gpu['name']}"))
            print(green(f"  Memory    : {gpu['vram_mb']} MB (unified)"))
        else:
            print(green(f"  GPU       : {gpu['name']}"))
            print(green(f"  VRAM      : {gpu['vram_mb']} MB"))
            print(green(f"  Driver    : {gpu['driver']}"))
    else:
        print(yellow("  GPU       : Algılanamadı — CPU modunda çalışılacak"))


def print_results_table(all_results):
    """Sonuçları model × test matrisi olarak yazdır."""
    print_header("BENCHMARK SONUÇLARI")

    for model, tests in all_results.items():
        print(bold(f"\n  ▸ {model}"))
        if "load_time" in tests:
            print(f"    Model yükleme süresi: {tests['load_time']}s")
        print()
        print(f"    {'Test':<30} {'Süre':>7} {'TTFT':>7} {'Token/s':>8} {'Token':>6} {'GPU%':>5} {'VRAM':>7}")
        print(f"    {'─'*30} {'─'*7} {'─'*7} {'─'*8} {'─'*6} {'─'*5} {'─'*7}")

        for tid in sorted(tests.keys()):
            if tid == "load_time":
                continue
            r = tests[tid]
            name = f"[{tid}] {r['test_name']}"
            ttft_str = f"{r['ttft_s']:.2f}s" if r.get('ttft_s') else "N/A"
            gpu_str = f"{r['avg_gpu_util']}%" if r['avg_gpu_util'] >= 0 else "N/A"
            vram_str = f"{r['max_vram_mb']}M" if r['max_vram_mb'] >= 0 else "N/A"
            print(
                f"    {name:<30} {r['total_time_s']:>6.1f}s {ttft_str:>7} "
                f"{r['tokens_per_sec']:>7.1f} {r['estimated_tokens']:>6} "
                f"{gpu_str:>5} {vram_str:>7}"
            )
    print()


def print_ranking(all_results):
    """Modelleri ortalama token/s'ye göre sırala."""
    print_header("MODEL SIRALAMASI (Ortalama Token/s)")

    scores = []
    for model, tests in all_results.items():
        tps_list = [tests[t]["tokens_per_sec"] for t in tests if t != "load_time"]
        if tps_list:
            avg = sum(tps_list) / len(tps_list)
            scores.append((model, round(avg, 1)))

    scores.sort(key=lambda x: x[1], reverse=True)
    for i, (model, avg_tps) in enumerate(scores, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ") if HAS_COLOR else f"{i}."
        bar_len = int(avg_tps / max(s[1] for s in scores) * 30) if scores else 0
        bar = "█" * bar_len
        print(f"  {medal} {model:<20} {avg_tps:>7.1f} tok/s  {green(bar)}")
    print()

# ── Dosya çıktıları ────────────────────────────────────────────────────────

def save_json(all_results, sys_info, gpu, filepath):
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": sys_info,
        "gpu": gpu,
        "results": {},
    }
    for model, tests in all_results.items():
        data["results"][model] = {}
        for tid, val in tests.items():
            data["results"][model][tid] = val
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(green(f"  ✓ JSON: {filepath}"))


def save_txt(all_results, sys_info, gpu, filepath):
    lines = []
    lines.append("=" * 70)
    lines.append("  OLLAMA BENCHMARK RAPORU")
    lines.append(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Sistem bilgisi
    lines.append("SİSTEM:")
    if sys_info.get("is_jetson"):
        lines.append(f"  Platform  : {sys_info.get('jetson_model', 'Jetson')}")
    else:
        lines.append(f"  Platform  : {sys_info['platform']} {sys_info['platform_release']}")
    lines.append(f"  CPU       : {sys_info.get('cpu_name', 'N/A')}")
    lines.append(f"  RAM       : {sys_info.get('ram_gb', 'N/A')} GB")
    if gpu["available"]:
        mem_label = "Memory (unified)" if gpu["type"] == "jetson" else "VRAM"
        lines.append(f"  GPU       : {gpu['name']}")
        lines.append(f"  {mem_label}: {gpu['vram_mb']} MB")
    lines.append("")

    # Sonuçlar
    for model, tests in all_results.items():
        lines.append("-" * 70)
        lines.append(f"MODEL: {model}")
        if "load_time" in tests:
            lines.append(f"  Yükleme süresi: {tests['load_time']}s")
        lines.append("")
        lines.append(f"  {'Test':<30} {'Süre':>7} {'TTFT':>7} {'Token/s':>8} {'Token':>6}")
        lines.append(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
        for tid in sorted(tests.keys()):
            if tid == "load_time":
                continue
            r = tests[tid]
            name = f"[{tid}] {r['test_name']}"
            ttft_str = f"{r['ttft_s']:.2f}s" if r.get('ttft_s') else "N/A"
            lines.append(
                f"  {name:<30} {r['total_time_s']:>6.1f}s {ttft_str:>7} "
                f"{r['tokens_per_sec']:>7.1f} {r['estimated_tokens']:>6}"
            )
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(green(f"  ✓ TXT: {filepath}"))


def save_csv(all_results, filepath):
    rows = []
    for model, tests in all_results.items():
        for tid in sorted(tests.keys()):
            if tid == "load_time":
                continue
            r = tests[tid]
            rows.append({
                "model": model,
                "test_id": tid,
                "test_name": r["test_name"],
                "total_time_s": r["total_time_s"],
                "ttft_s": r.get("ttft_s", ""),
                "tokens_per_sec": r["tokens_per_sec"],
                "estimated_tokens": r["estimated_tokens"],
                "eval_count": r.get("eval_count", ""),
                "prompt_eval_ms": r.get("prompt_eval_ms", ""),
                "word_count": r["word_count"],
                "avg_gpu_util": r["avg_gpu_util"] if r["avg_gpu_util"] >= 0 else "",
                "max_vram_mb": r["max_vram_mb"] if r["max_vram_mb"] >= 0 else "",
                "response_length": r["response_length"],
            })

    if rows:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(green(f"  ✓ CSV: {filepath}"))

# ── Ana akış ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama LLM Benchmark Tool")
    parser.add_argument("--skip-download", action="store_true", help="Model indirmeyi atla")
    parser.add_argument("--models", type=str, default=None, help='Belirli modeller: "gemma3:4b,qwen3:4b"')
    parser.add_argument("--quick", action="store_true", help="Sadece Test B çalıştır")
    parser.add_argument("--output", type=str, default="benchmark", help="Çıktı dosya prefiksi")
    args = parser.parse_args()

    print_header("OLLAMA LLM BENCHMARK")
    print(f"  Tarih : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Hedef : {'Hızlı mod (Test B)' if args.quick else 'Tam benchmark (A-E)'}")
    print()

    # 1. Sistem bilgisi
    print(cyan("● Sistem bilgileri toplanıyor..."))
    sys_info = get_system_info()
    gpu = get_gpu_info()
    print_system_report(sys_info, gpu)

    # 2. Ollama kontrolü
    print(cyan("\n● Ollama kontrol ediliyor..."))
    check_ollama()

    # 3. Model listesi
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS[:]

    # 4. Modelleri hazırla
    print(cyan("\n● Modeller hazırlanıyor..."))
    ready_models = ensure_models(models, args.skip_download)

    if not ready_models:
        print(red("\n✗ Hiçbir model hazır değil. Çıkılıyor."))
        sys.exit(1)

    print(green(f"\n  Hazır modeller: {', '.join(ready_models)}"))

    # 5. Hangi testler?
    if args.quick:
        test_ids = ["B"]
    else:
        test_ids = sorted(BENCHMARKS.keys())

    # 6. Benchmark!
    all_results = {}
    total_tests = len(ready_models) * len(test_ids)
    done_tests = 0

    for model in ready_models:
        if interrupted:
            break

        print_header(f"MODEL: {model}")
        all_results[model] = {}

        # Warmup — model yükleme süresi
        load_time = warmup_model(model)
        all_results[model]["load_time"] = load_time

        for tid in test_ids:
            if interrupted:
                break

            bench = BENCHMARKS[tid]
            done_tests += 1
            print(bold(f"\n  ▶ Test {tid}: {bench['name']}  [{done_tests}/{total_tests}]"))
            print(f"    Prompt: {bench['prompt'][:80]}...")
            print(cyan("    Yanıt: "), end="")

            result = run_benchmark(model, tid, bench["prompt"])
            all_results[model][tid] = result

            print()
            print(
                f"    → {result['total_time_s']}s | "
                f"{result['tokens_per_sec']} tok/s | "
                f"TTFT: {result.get('ttft_s', 'N/A')}s | "
                f"Tokens: {result['estimated_tokens']}"
            )
            if result["avg_gpu_util"] >= 0:
                print(f"    → GPU: {result['avg_gpu_util']}% | VRAM: {result['max_vram_mb']} MB")

    # 7. Rapor
    if all_results:
        print_results_table(all_results)
        print_ranking(all_results)

        # Dosya çıktıları
        print(cyan("● Sonuçlar kaydediliyor..."))
        prefix = args.output
        save_json(all_results, sys_info, gpu, f"{prefix}_results.json")
        save_txt(all_results, sys_info, gpu, f"{prefix}_report.txt")
        save_csv(all_results, f"{prefix}_report.csv")
        print()

    print(bold("✓ Benchmark tamamlandı!"))


if __name__ == "__main__":
    main()
