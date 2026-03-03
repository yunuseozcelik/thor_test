#!/usr/bin/env python3
# SAM3 Video Benchmark - Canli Overlay + FPS Olcum
#
# Kullanim:
#   python sam_video_benchmark.py
#   python sam_video_benchmark.py --model sam3.pt --video E:/test/sam_test_videos/ZAHA.mp4
#   python sam_video_benchmark.py --no-display
#   python sam_video_benchmark.py --save-video out.mp4
#   python sam_video_benchmark.py --quick

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
from collections import deque

try:
    import torch
    import numpy as np
    import cv2
except ImportError as e:
    print(f"Eksik kutuphane: {e}")
    print("Kurulum: pip install torch torchvision numpy opencv-python")
    sys.exit(1)

# segment_anything opsiyonel — varsa gercek SAM inference yapar
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    pass

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    CC = True
except ImportError:
    CC = False

def green(t):  return f"{Fore.GREEN}{t}{Style.RESET_ALL}" if CC else str(t)
def red(t):    return f"{Fore.RED}{t}{Style.RESET_ALL}" if CC else str(t)
def yellow(t): return f"{Fore.YELLOW}{t}{Style.RESET_ALL}" if CC else str(t)
def cyan(t):   return f"{Fore.CYAN}{t}{Style.RESET_ALL}" if CC else str(t)
def bold(t):   return f"{Fore.WHITE}{Style.BRIGHT}{t}{Style.RESET_ALL}" if CC else str(t)

# ── Global ──────────────────────────────────────────────────────────────────

interrupted = False

def _sig(s, f):
    global interrupted
    if not interrupted:
        print(yellow("\n\n  Durduruluyor... (sonuclar kaydedilecek)"))
    interrupted = True

signal.signal(signal.SIGINT, _sig)

def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception:
        return -1, "", ""

def is_jetson():
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/device-tree/model") as f:
            return "jetson" in f.read().lower()
    except Exception:
        return False

# ── Sistem bilgisi ──────────────────────────────────────────────────────────

def get_system_info():
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "opencv": cv2.__version__,
        "is_jetson": is_jetson(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    info["cpu_cores"] = os.cpu_count() or 0
    return info

# ── GPU izleme ──────────────────────────────────────────────────────────────

class GpuMonitor:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()

    def start(self):
        self.samples = []
        self._stop.clear()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            if torch.cuda.is_available():
                self.samples.append({
                    "vram_mb": round(torch.cuda.memory_allocated(0) / 1024**2),
                    "reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2),
                })
            self._stop.wait(0.5)

    def summary(self):
        if not self.samples:
            return {}
        vrams = [s["vram_mb"] for s in self.samples]
        return {
            "avg_vram_mb": round(sum(vrams) / len(vrams)),
            "peak_vram_mb": max(vrams),
        }

# ── SAM model tipi tahmini ─────────────────────────────────────────────────

def _guess_sam_type_from_state(state_dict):
    """State dict'teki katman sayisina gore SAM model tipini tahmin et."""
    # SAM ViT-H: ~1400+ layers, ~636M encoder params
    # SAM ViT-L: ~800+ layers, ~307M encoder params
    # SAM ViT-B: ~400+ layers, ~89M encoder params
    n_layers = len([k for k in state_dict if isinstance(state_dict[k], torch.Tensor)])
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))

    if total_params > 600_000_000 or n_layers > 1200:
        return "vit_h"
    elif total_params > 200_000_000 or n_layers > 600:
        return "vit_l"
    else:
        return "vit_b"


def _guess_sam_type_from_size(size_mb):
    """Dosya boyutundan SAM tipini tahmin et."""
    if size_mb > 2000:
        return "vit_h"    # ~2.5GB
    elif size_mb > 1000:
        return "vit_l"    # ~1.2GB
    else:
        return "vit_b"    # ~375MB

# ── Model yukleme ──────────────────────────────────────────────────────────

def load_model(path, device, sam_points_per_side=16):
    path = Path(path)
    size_mb = path.stat().st_size / 1024**2
    print(f"  Dosya  : {path.name} ({size_mb:.0f} MB)")
    print(f"  Device : {device}")

    # 1) TorchScript
    print(cyan("  -> TorchScript..."), end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        dt = time.perf_counter() - t0
        print(green(f"OK ({dt:.2f}s)"))
        return {
            "model": model, "type": "torchscript", "load_time": round(dt, 2),
            "size_mb": round(size_mb), "sam_mode": "torchscript",
        }
    except Exception as e:
        print(yellow(f"Degil ({type(e).__name__})"))

    # 2) torch.load
    print(cyan("  -> torch.load..."), end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        obj = torch.load(str(path), map_location=device, weights_only=False)
        dt = time.perf_counter() - t0

        if isinstance(obj, torch.nn.Module):
            obj.eval()
            params = sum(p.numel() for p in obj.parameters())
            print(green(f"nn.Module ({dt:.2f}s, {params:,} params)"))
            return {
                "model": obj, "type": "nn_module", "load_time": round(dt, 2),
                "size_mb": round(size_mb), "params": params, "sam_mode": "full_model",
            }

        if isinstance(obj, dict):
            print(yellow("Dict (checkpoint)"))
            return _load_sam_from_checkpoint(obj, dt, size_mb, path, device, sam_points_per_side)

    except Exception as e:
        print(red(f"Hata: {e}"))

    return None


def _load_sam_from_checkpoint(ckpt, load_time, size_mb, path, device, sam_points_per_side=16):
    """SAM checkpoint'ini yukle — segment_anything varsa gercek model, yoksa analiz."""
    # State dict bul
    state = None
    for key in ["model", "state_dict", "model_state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None and all(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:5]):
        state = ckpt

    if state is None:
        print(red("    State dict bulunamadi"))
        return None

    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    total_params = sum(v.numel() for v in tensors.values())
    n_layers = len(tensors)

    # SAM mi kontrol et
    is_sam = any("image_encoder" in k or "mask_decoder" in k or "prompt_encoder" in k for k in tensors)
    sam_type = _guess_sam_type_from_state(state) if is_sam else _guess_sam_type_from_size(size_mb)

    print(f"    Parametreler : {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"    Katmanlar    : {n_layers}")
    if is_sam:
        print(f"    SAM tipi     : {sam_type} (otomatik tespit)")

    info = {
        "type": "sam_checkpoint",
        "load_time": round(load_time, 2),
        "size_mb": round(size_mb),
        "params": total_params,
        "n_layers": n_layers,
        "sam_type": sam_type,
        "is_sam": is_sam,
    }

    # ── segment_anything ile gercek model yukleme ──
    if SAM_AVAILABLE:
        print(cyan(f"    segment_anything ile yukleniyor ({sam_type})..."), end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            sam = sam_model_registry[sam_type](checkpoint=str(path))
            sam.to(device=device)
            sam.eval()
            dt = time.perf_counter() - t0
            print(green(f"OK ({dt:.2f}s)"))

            predictor = SamPredictor(sam)
            mask_gen = SamAutomaticMaskGenerator(
                sam,
                points_per_side=max(4, int(sam_points_per_side)),  # hiz/kalite ayari (varsayilan 32)
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=100,
            )

            info["model"] = sam
            info["predictor"] = predictor
            info["mask_generator"] = mask_gen
            info["sam_mode"] = "full_sam"
            info["load_time"] = round(load_time + dt, 2)
            return info

        except Exception as e:
            print(red(f"Hata: {e}"))

    # ── segment_anything yok — sadece image encoder yukle ──
    if SAM_AVAILABLE:
        print(yellow("    SAM tam model yuklenemedi, image encoder deneniyor..."))
    else:
        print(yellow("    segment_anything kurulu degil"))
        print(yellow("    Kurulum: pip install git+https://github.com/facebookresearch/segment-anything.git"))
        print(yellow("    Kurulunca gercek segmentasyon overlay gorebilirsiniz"))
        print(yellow("    Simdilik sadece image encoder throughput testi yapilacak"))

    # Image encoder'i state dict'ten soyutla
    encoder_state = {k.replace("image_encoder.", ""): v for k, v in state.items() if k.startswith("image_encoder.")}
    if not encoder_state:
        # SAM olmayabilir — tum state'i kullan
        encoder_state = state

    # Basit ViT-benzeri encoder olustur (gercek boyutlarda)
    # SAM image encoder: patch_embed -> blocks -> neck
    info["model"] = _build_encoder_proxy(encoder_state, device)
    info["sam_mode"] = "encoder_proxy"
    return info


def _build_encoder_proxy(encoder_state, device):
    """
    State dict'in boyutlarindan turetilen bir encoder proxy.
    Gercek SAM encoder degil ama ayni hesaplama yukunu simule eder.
    """
    # patch_embed.proj: [embed_dim, 3, kernel, kernel]
    embed_dim = 256  # varsayilan
    for k, v in encoder_state.items():
        if "patch_embed" in k and "proj" in k and v.dim() == 4:
            embed_dim = v.shape[0]
            break

    # Block sayisi
    block_indices = set()
    for k in encoder_state:
        m = re.search(r"blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    n_blocks = len(block_indices) if block_indices else 12

    print(f"    Encoder: embed_dim={embed_dim}, blocks={n_blocks}")

    class EncoderProxy(torch.nn.Module):
        """SAM image encoder ile benzer hesaplama yuku."""
        def __init__(self, embed_dim, n_blocks):
            super().__init__()
            self.patch_embed = torch.nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            # Transformer bloklari simule et
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8 if embed_dim >= 512 else 4,
                dim_feedforward=embed_dim * 4, batch_first=True,
                dropout=0.0,
            )
            self.blocks = torch.nn.TransformerEncoder(encoder_layer, num_layers=min(n_blocks, 32))
            self.neck = torch.nn.Conv2d(embed_dim, 256, 1)

        def forward(self, x):
            # x: [B, 3, H, W]
            x = self.patch_embed(x)           # [B, embed_dim, H/16, W/16]
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
            x = self.blocks(x)                # [B, H*W, embed_dim]
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.neck(x)                  # [B, 256, H/16, W/16]
            return x

    model = EncoderProxy(embed_dim, n_blocks).to(device).eval()
    return model

# ── Overlay cizim ───────────────────────────────────────────────────────────

def masks_to_overlay(frame, masks_data):
    """SAM mask sonuclarini renkli overlay'e cevir."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if not masks_data:
        return overlay

    # Her mask icin rastgele renk
    np.random.seed(0)
    colors = np.random.randint(80, 255, (len(masks_data), 3))

    # Alana gore sirala — buyukten kucuge (buyukler altta)
    sorted_masks = sorted(masks_data, key=lambda x: x["area"], reverse=True)

    for i, mask_info in enumerate(sorted_masks):
        mask = mask_info["segmentation"]  # bool array [H, W]
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.35, 0)

        # Mask kenari ciz
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1)

    return overlay


def tensor_to_heatmap_overlay(frame, output_tensor):
    """Model ciktisini heatmap overlay'e cevir (SAM olmayan modeller icin)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if output_tensor is None:
        return overlay

    try:
        out = output_tensor.detach().cpu()
        while out.dim() > 2:
            out = out.mean(dim=0) if out.dim() > 3 else out.squeeze(0)
        if out.dim() < 2:
            return overlay

        out = out.float()
        mn, mx = out.min(), out.max()
        if mx > mn:
            out = ((out - mn) / (mx - mn) * 255).byte().numpy()
        else:
            return overlay

        heatmap = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.addWeighted(colored, 0.4, overlay, 0.6, 0, overlay)
    except Exception:
        pass

    return overlay


def draw_hud(frame, frame_idx, fps, avg_ms, vram_mb, n_masks, sam_mode):
    """Performans HUD ciz."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Ust bar
    bar_h = 50
    sub = overlay[0:bar_h, :]
    dark = (sub * 0.3).astype(np.uint8)
    overlay[0:bar_h, :] = dark

    font = cv2.FONT_HERSHEY_SIMPLEX

    # FPS
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 32), font, 0.8, fps_color, 2)

    # Latency
    cv2.putText(overlay, f"{avg_ms:.1f}ms", (180, 32), font, 0.8, (0, 255, 255), 2)

    # Frame
    cv2.putText(overlay, f"#{frame_idx}", (330, 32), font, 0.7, (255, 255, 255), 2)

    # Mask sayisi
    if n_masks >= 0:
        cv2.putText(overlay, f"Masks: {n_masks}", (430, 32), font, 0.7, (255, 180, 0), 2)

    # VRAM
    if vram_mb > 0:
        cv2.putText(overlay, f"VRAM:{vram_mb}MB", (w - 180, 32), font, 0.6, (100, 200, 255), 2)

    # Mode indicator
    mode_text = {"full_sam": "SAM", "encoder_proxy": "PROXY", "torchscript": "JIT", "full_model": "MODEL"}.get(sam_mode, "?")
    mode_color = (0, 255, 0) if sam_mode == "full_sam" else (0, 200, 255)
    cv2.putText(overlay, mode_text, (w - 60, 32), font, 0.5, mode_color, 2)

    return overlay


def draw_fps_graph(overlay, fps_history, x_start, y_start, graph_w, graph_h):
    """Kucuk FPS grafigi ciz."""
    if len(fps_history) < 2:
        return overlay

    cv2.rectangle(overlay, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (80, 80, 80), 1)

    fps_list = list(fps_history)
    max_fps = max(max(fps_list), 1)
    min_fps = min(fps_list)

    n = len(fps_list)
    step_x = graph_w / max(n - 1, 1)
    pts = []
    for i, f in enumerate(fps_list):
        x = int(x_start + i * step_x)
        ratio = (f - min_fps) / (max_fps - min_fps) if max_fps > min_fps else 0.5
        y = int(y_start + graph_h - ratio * (graph_h - 10) - 5)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        ratio = fps_list[i+1] / max_fps if max_fps > 0 else 0
        color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        cv2.line(overlay, pts[i], pts[i+1], color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"{max_fps:.0f}", (x_start + 2, y_start + 14), font, 0.4, (150, 150, 150), 1)
    cv2.putText(overlay, f"{min_fps:.0f}", (x_start + 2, y_start + graph_h - 4), font, 0.4, (150, 150, 150), 1)

    return overlay

# ── Video benchmark ─────────────────────────────────────────────────────────


def _resize_frame_for_sam(frame, target_res):
    """
    SAM icin frame'i hiz amacli kucult.
    Aspect ratio korunur, max kenar target_res olur.
    """
    if target_res <= 0:
        return frame, False
    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side <= target_res:
        return frame, False
    scale = float(target_res) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, True


def _restore_masks_to_frame_size(masks_data, out_h, out_w):
    """Kucuk frame uzerinde uretilen masklari orijinal frame boyutuna geri tasir."""
    restored = []
    for mask_info in masks_data:
        seg = mask_info.get("segmentation")
        if seg is None:
            continue
        seg_u8 = seg.astype(np.uint8)
        seg_resized = cv2.resize(seg_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        item = dict(mask_info)
        item["segmentation"] = seg_resized
        item["area"] = int(seg_resized.sum())
        restored.append(item)
    return restored


def run_video_benchmark(model_info, video_path, device, args):
    """Video uzerinde frame-by-frame benchmark + canli gosterim."""
    sam_mode = model_info.get("sam_mode", "unknown")
    model = model_info.get("model")
    mask_generator = model_info.get("mask_generator")

    use_sam = (sam_mode == "full_sam" and mask_generator is not None)

    if model is None and not use_sam:
        print(red("  Model forward pass desteklemiyor"))
        return None

    target_res = args.resolution

    # Video ac
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(red(f"  X Video acilamadi: {video_path}"))
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    print(f"  Video    : {Path(video_path).name}")
    print(f"  Boyut    : {src_w}x{src_h} @ {src_fps:.1f} FPS")
    print(f"  Sure     : {duration:.1f}s ({total_frames} kare)")
    if use_sam:
        print(green(f"  Mod      : SAM segmentasyon (gercek mask overlay)"))
        print(f"  SAM giris: max_side={target_res}")
    else:
        print(yellow(f"  Mod      : Encoder proxy (heatmap overlay)"))
        print(f"  Giris    : {target_res}x{target_res}")

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    show_display = args.display

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, src_fps, (src_w, src_h))
        print(f"  Kayit    : {args.save_video}")

    # Metrikler
    frame_times = []
    total_times = []
    mask_counts = []
    fps_history = deque(maxlen=120)
    gpu_mon = GpuMonitor()
    gpu_mon.start()

    # Warmup
    print(cyan("\n  Warmup..."), end=" ", flush=True)
    ret, warmup_frame = cap.read()
    if ret:
        try:
            if use_sam:
                sam_input, _ = _resize_frame_for_sam(warmup_frame, target_res)
                rgb = cv2.cvtColor(sam_input, cv2.COLOR_BGR2RGB)
                _ = mask_generator.generate(rgb)
            else:
                tensor = _frame_to_tensor(warmup_frame, target_res, device)
                with torch.no_grad():
                    _ = model(tensor)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
        except Exception as e:
            print(yellow(f"warmup hata: {e}"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(green("OK"))

    # Ana dongu
    print(cyan("  Basliyor... (ESC/Q ile cik)\n"))
    frame_idx = 0
    t_loop_start = time.perf_counter()

    while not interrupted:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        t_frame_start = time.perf_counter()
        n_masks = -1
        output_tensor = None

        if use_sam:
            # ── Gercek SAM segmentasyon ──
            sam_input, resized_for_sam = _resize_frame_for_sam(frame, target_res)
            rgb = cv2.cvtColor(sam_input, cv2.COLOR_BGR2RGB)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = time.perf_counter()

            masks_data = mask_generator.generate(rgb)
            if resized_for_sam:
                masks_data = _restore_masks_to_frame_size(masks_data, frame.shape[0], frame.shape[1])

            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_ms = (time.perf_counter() - t_inf) * 1000

            n_masks = len(masks_data)
            mask_counts.append(n_masks)

            # Overlay — gercek masklar
            display_frame = masks_to_overlay(frame, masks_data)

        else:
            # ── Encoder proxy / genel model ──
            tensor = _frame_to_tensor(frame, target_res, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = time.perf_counter()

            with torch.no_grad():
                output_tensor = model(tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_ms = (time.perf_counter() - t_inf) * 1000

            # Overlay — heatmap
            display_frame = tensor_to_heatmap_overlay(frame, output_tensor)

        frame_times.append(inf_ms)

        # FPS
        current_fps = 1000.0 / inf_ms if inf_ms > 0 else 0
        fps_history.append(current_fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # VRAM
        vram = round(torch.cuda.memory_allocated(0) / 1024**2) if torch.cuda.is_available() else -1

        # HUD
        display_frame = draw_hud(display_frame, frame_idx, avg_fps, inf_ms, vram, n_masks, sam_mode)

        # FPS grafigi
        graph_w, graph_h = 200, 60
        display_frame = draw_fps_graph(
            display_frame, fps_history,
            src_w - graph_w - 10, src_h - graph_h - 10,
            graph_w, graph_h
        )

        # Goster
        if show_display:
            cv2.imshow("SAM3 Benchmark", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                print(yellow("\n  Kullanici durdurdu"))
                break

        if writer:
            writer.write(display_frame)

        t_frame_end = time.perf_counter()
        total_times.append((t_frame_end - t_frame_start) * 1000)

        # Terminal progress
        if frame_idx % 30 == 0:
            pct = frame_idx / max_frames * 100
            filled = int(25 * pct / 100)
            bar = "#" * filled + "-" * (25 - filled)
            elapsed = time.perf_counter() - t_loop_start
            mask_str = f"| Masks: {n_masks}" if n_masks >= 0 else ""
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}% | Frame {frame_idx}/{max_frames} | "
                f"FPS: {avg_fps:.1f} | {inf_ms:.1f}ms {mask_str} | {elapsed:.0f}s  "
            )
            sys.stdout.flush()

        frame_idx += 1

    # Temizlik
    total_elapsed = time.perf_counter() - t_loop_start
    gpu_mon.stop()
    cap.release()
    if writer:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()

    print(f"\n\n  Islenen kare: {frame_idx}")

    if not frame_times:
        return None

    ft = np.array(frame_times)
    tt = np.array(total_times)
    gpu_s = gpu_mon.summary()

    results = {
        "video": str(video_path),
        "video_resolution": f"{src_w}x{src_h}",
        "video_fps": src_fps,
        "video_duration_s": round(duration, 1),
        "total_frames_processed": frame_idx,
        "model_input_resolution": target_res,
        "sam_mode": sam_mode,
        "total_elapsed_s": round(total_elapsed, 1),
        "inference": {
            "avg_ms": round(float(ft.mean()), 2),
            "std_ms": round(float(ft.std()), 2),
            "min_ms": round(float(ft.min()), 2),
            "max_ms": round(float(ft.max()), 2),
            "median_ms": round(float(np.median(ft)), 2),
            "p95_ms": round(float(np.percentile(ft, 95)), 2),
            "p99_ms": round(float(np.percentile(ft, 99)), 2),
            "avg_fps": round(1000.0 / float(ft.mean()), 1) if ft.mean() > 0 else 0,
        },
        "total_pipeline": {
            "avg_ms": round(float(tt.mean()), 2),
            "avg_fps": round(1000.0 / float(tt.mean()), 1) if tt.mean() > 0 else 0,
        },
        "gpu": gpu_s,
        "realtime_capable": float(ft.mean()) < (1000.0 / src_fps) if src_fps > 0 else False,
    }

    if mask_counts:
        results["masks"] = {
            "avg": round(np.mean(mask_counts), 1),
            "min": int(np.min(mask_counts)),
            "max": int(np.max(mask_counts)),
        }

    return results


def _frame_to_tensor(frame, target_res, device):
    """OpenCV BGR frame -> model input tensor."""
    resized = cv2.resize(frame, (target_res, target_res), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

# ── Rapor ───────────────────────────────────────────────────────────────────

def print_report(results, model_info, sys_info):
    w = 65
    print("\n" + bold("=" * w))
    print(bold("  SAM3 VIDEO BENCHMARK SONUCLARI"))
    print(bold("=" * w))

    print(f"  Video          : {Path(results['video']).name}")
    print(f"  Video boyutu   : {results['video_resolution']} @ {results['video_fps']} FPS")
    print(f"  Islenen kare   : {results['total_frames_processed']}")
    print(f"  Toplam sure    : {results['total_elapsed_s']}s")
    print(f"  Model          : {model_info.get('sam_type', model_info['type'])} ({model_info.get('size_mb', '?')} MB)")
    print(f"  Mod            : {results['sam_mode']}")
    if sys_info.get("gpu_name"):
        print(f"  GPU            : {sys_info['gpu_name']}")

    inf = results["inference"]
    pipe = results["total_pipeline"]
    print()
    print(bold("  Inference:"))
    print(f"    Ortalama     : {inf['avg_ms']:.2f}ms")
    print(f"    Std          : {inf['std_ms']:.2f}ms")
    print(f"    Min          : {inf['min_ms']:.2f}ms")
    print(f"    Max          : {inf['max_ms']:.2f}ms")
    print(f"    Median       : {inf['median_ms']:.2f}ms")
    print(f"    P95          : {inf['p95_ms']:.2f}ms")
    print(f"    P99          : {inf['p99_ms']:.2f}ms")
    print(green(f"    FPS          : {inf['avg_fps']}"))

    print()
    print(bold("  Pipeline (inference + overlay + display):"))
    print(f"    Ortalama     : {pipe['avg_ms']:.2f}ms")
    print(green(f"    FPS          : {pipe['avg_fps']}"))

    if "masks" in results:
        print()
        print(bold("  Segmentasyon:"))
        m = results["masks"]
        print(f"    Ort. mask    : {m['avg']}")
        print(f"    Min/Max      : {m['min']} / {m['max']}")

    if results.get("gpu"):
        print()
        print(bold("  GPU:"))
        for k, v in results["gpu"].items():
            print(f"    {k}: {v}")

    # Gercek zamanli mi?
    print()
    rt = results["realtime_capable"]
    needed_ms = 1000.0 / results["video_fps"] if results["video_fps"] > 0 else 33.3
    if rt:
        print(green(f"  GERCEK ZAMANLI: {inf['avg_ms']:.1f}ms < {needed_ms:.1f}ms (video {results['video_fps']} FPS)"))
    else:
        print(yellow(f"  Gercek zamanli DEGIL: {inf['avg_ms']:.1f}ms > {needed_ms:.1f}ms gerekli"))
        if inf['avg_ms'] > 0:
            ratio = inf['avg_ms'] / needed_ms
            print(yellow(f"    {ratio:.1f}x yavas"))

    print()


def save_results(results, model_info, sys_info, prefix):
    # JSON
    jpath = f"{prefix}_video_results.json"
    data = {
        "timestamp": datetime.now().isoformat(),
        "system": sys_info,
        "model": {k: v for k, v in model_info.items()
                  if k not in ("model", "predictor", "mask_generator", "state_dict", "checkpoint")},
        "results": results,
    }
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(green(f"  {jpath}"))

    # CSV
    cpath = f"{prefix}_video_results.csv"
    flat = {"video": results["video"], "resolution": results["video_resolution"],
            "sam_mode": results["sam_mode"]}
    flat.update({f"inf_{k}": v for k, v in results["inference"].items()})
    flat.update({f"pipe_{k}": v for k, v in results["total_pipeline"].items()})
    flat.update(results.get("gpu", {}))
    flat["realtime"] = results["realtime_capable"]
    if "masks" in results:
        flat.update({f"mask_{k}": v for k, v in results["masks"].items()})
    with open(cpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat.keys())
        w.writeheader()
        w.writerow(flat)
    print(green(f"  {cpath}"))

    # TXT
    tpath = f"{prefix}_video_report.txt"
    inf = results["inference"]
    lines = [
        "=" * 60,
        "SAM3 VIDEO BENCHMARK",
        f"Tarih: {datetime.now()}",
        f"Video: {results['video']} ({results['video_resolution']})",
        f"Model: {model_info.get('sam_type', model_info['type'])} | {model_info.get('size_mb', '?')} MB",
        f"Mod: {results['sam_mode']}",
        f"GPU: {sys_info.get('gpu_name', 'CPU')}",
        "=" * 60,
        f"Kareler: {results['total_frames_processed']}",
        f"Inference: {inf['avg_ms']}ms avg | {inf['avg_fps']} FPS",
        f"P95: {inf['p95_ms']}ms | P99: {inf['p99_ms']}ms",
        f"Pipeline: {results['total_pipeline']['avg_ms']}ms | {results['total_pipeline']['avg_fps']} FPS",
        f"Gercek zamanli: {'EVET' if results['realtime_capable'] else 'HAYIR'}",
    ]
    if "masks" in results:
        lines.append(f"Masklar: ort={results['masks']['avg']} min={results['masks']['min']} max={results['masks']['max']}")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(green(f"  {tpath}"))

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3 Video Benchmark")
    parser.add_argument("--model", default="sam3.pt", help="Model dosyasi")
    parser.add_argument("--video", default=r"E:\test\sam_test_videos\ZAHA.mp4", help="Video dosyasi")
    parser.add_argument("--resolution", type=int, default=512, help="Model giris cozunurlugu / SAM max_side")
    parser.add_argument("--sam-points-per-side", type=int, default=8, help="SAM mask noktasi (dusuk=hizli, yuksek=kaliteli)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maks kare (0=tumu)")
    parser.add_argument("--quick", action="store_true", help="Ilk 100 kare")
    parser.add_argument("--no-display", dest="display", action="store_false", help="Canli gosterim kapali")
    parser.add_argument("--save-video", type=str, default=None, help="Cikti video (orn: out.mp4)")
    parser.add_argument("--output", default="benchmark", help="Dosya prefiksi")
    parser.add_argument("--cpu", action="store_true", help="CPU modu")
    parser.set_defaults(display=True)
    args = parser.parse_args()

    if args.quick:
        args.max_frames = 100

    print(bold("=" * 60))
    print(bold("  SAM3 VIDEO BENCHMARK"))
    print(bold(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(bold("=" * 60))

    # Sistem
    print(cyan("\n  Sistem..."))
    sys_info = get_system_info()
    if sys_info.get("gpu_name"):
        print(f"  {sys_info['gpu_name']} | {sys_info.get('gpu_vram_mb', '?')} MB VRAM")
    else:
        print(yellow("  GPU yok — CPU"))
    print(f"  segment_anything: {'EVET' if SAM_AVAILABLE else 'HAYIR'}")

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"  Device: {device}")

    # Video kontrol
    video_path = Path(args.video)
    if not video_path.is_file():
        print(red(f"\n  X Video bulunamadi: {args.video}"))
        for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            for p in Path(".").rglob(ext):
                print(f"    {p}")
        sys.exit(1)

    # Model
    print(cyan("\n  Model yukleniyor..."))
    model_path = Path(args.model)
    if not model_path.is_file():
        pts = list(Path(".").glob("*.pt")) + list(Path(".").glob("*.pth"))
        print(red(f"  X '{args.model}' bulunamadi"))
        if pts:
            print("  Mevcut modeller:")
            for p in pts:
                print(f"    {p.name} ({p.stat().st_size/1024**2:.0f} MB)")
        sys.exit(1)

    model_info = load_model(model_path, device, sam_points_per_side=args.sam_points_per_side)
    if model_info is None:
        print(red("  Model yuklenemedi"))
        sys.exit(1)

    sam_mode = model_info.get("sam_mode", "unknown")
    print(f"  Mod: {sam_mode}")

    # Benchmark
    print(cyan("\n  Video benchmark basliyor..."))
    results = run_video_benchmark(model_info, video_path, device, args)

    if results:
        print_report(results, model_info, sys_info)
        print(cyan("  Kaydediliyor..."))
        save_results(results, model_info, sys_info, args.output)

    print(bold("\n  Tamamlandi!"))


if __name__ == "__main__":
    main()
