#!/usr/bin/env python3
"""
Audio Visualizer - Generate MP4 FFT spectrum visualization from audio files.
Usage: python3 visualizer.py input.m4a -o output.mp4
"""

import argparse
import colorsys
import math
import os
import random
import subprocess
import sys
from multiprocessing import Pool, cpu_count
from multiprocessing.shared_memory import SharedMemory

import librosa
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Module-level globals — populated by _worker_init() in each worker process
# ---------------------------------------------------------------------------
_shm = None           # SharedMemory handle (kept open so OS doesn't reclaim it)
_spectrum = None      # numpy view into shared memory: (n_frames, n_bars) float32
_bar_colors = None    # (n_bars, 3) uint8
_bg_color = None      # (R, G, B) tuple
_font = None          # ImageFont for labels
_font_small = None    # smaller ImageFont
_W = None             # frame width
_H = None             # frame height
_n_frames = None
_n_bars = None
_duration = None
_fps = None
_filename = None


# ---------------------------------------------------------------------------
# Worker initializer (called once per worker process on Pool start)
# ---------------------------------------------------------------------------
def _worker_init(shm_name, shm_shape, shm_dtype,
                 bar_colors, bg_color, W, H,
                 n_frames, n_bars, duration, fps, filename):
    global _shm, _spectrum, _bar_colors, _bg_color, _font, _font_small
    global _W, _H, _n_frames, _n_bars, _duration, _fps, _filename

    _shm = SharedMemory(name=shm_name)
    _spectrum = np.ndarray(shm_shape, dtype=shm_dtype, buffer=_shm.buf)

    _bar_colors = bar_colors
    _bg_color = bg_color
    _W = W
    _H = H
    _n_frames = n_frames
    _n_bars = n_bars
    _duration = duration
    _fps = fps
    _filename = filename

    # Try to load a monospace font; fall back to PIL default
    font_paths = [
        '/System/Library/Fonts/Menlo.ttc',
        '/System/Library/Fonts/Monaco.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/TTF/DejaVuSansMono.ttf',
    ]
    _font = None
    _font_small = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                _font = ImageFont.truetype(fp, 18, index=0)
                _font_small = ImageFont.truetype(fp, 14, index=0)
                break
            except Exception:
                continue
    if _font is None:
        _font = ImageFont.load_default()
        _font_small = _font


# ---------------------------------------------------------------------------
# Frame renderer (called in worker processes)
# ---------------------------------------------------------------------------
def render_frame(frame_idx):
    bars = _spectrum[frame_idx]          # (n_bars,) float32, values in [0, 1]

    W, H = _W, _H
    bg = _bg_color

    img = Image.new('RGB', (W, H), bg)
    draw = ImageDraw.Draw(img)

    # --- Bar geometry ---
    # Bars fill each half symmetrically from the horizontal center outward.
    margin_x = int(W * 0.04)
    center_x = W // 2
    half_area_w = center_x - margin_x        # available pixels for n_bars in each half
    bar_gap = max(2, half_area_w // (_n_bars * 8))
    bar_w = (half_area_w - bar_gap * (_n_bars - 1)) // _n_bars

    progress_bar_h = 12
    progress_margin = int(H * 0.04)
    label_area_h = 40                        # space at top for filename label

    usable_top = label_area_h
    usable_bottom = H - progress_margin - progress_bar_h - 20
    usable_h = usable_bottom - usable_top
    center_y = usable_top + int(usable_h * 0.70)  # 중심선 20% 아래로 이동 (50% → 70%)
    up_max = int(usable_h * 0.585)   # 위쪽 최대 픽셀 (65% × 0.9)
    down_max = int(usable_h * 0.315) # 아래쪽 최대 픽셀 (35% × 0.9)

    # i=0: single centered bar.  i>=1: paired bars, offset past the center bar + gap.
    half_bar = bar_w // 2

    def center_rect():
        """Single bar centered on center_x."""
        x0 = center_x - half_bar
        return x0, x0 + bar_w - 1

    def bar_rects(i):
        """Return (rx0, rx1, lx0, lx1) for side bar i >= 1."""
        offset = (bar_w - half_bar) + bar_gap + (i - 1) * (bar_w + bar_gap)
        rx0 = center_x + offset
        rx1 = rx0 + bar_w - 1
        lx1 = center_x - 1 - offset
        lx0 = lx1 - bar_w + 1
        return rx0, rx1, lx0, lx1

    # --- Draw glow layer (down-sampled blur) ---
    glow_scale = 4
    glow_w, glow_h = W // glow_scale, H // glow_scale
    glow_img = Image.new('RGB', (glow_w, glow_h), (0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_img)

    for i in range(_n_bars):
        h_val = bars[i]
        if h_val < 0.005:
            continue
        up_h = max(1, int(h_val * up_max))
        down_h = max(1, int(h_val * down_max))
        color = tuple(_bar_colors[i])
        gy_top = (center_y - up_h) // glow_scale
        gy_bot = (center_y + down_h) // glow_scale

        if i == 0:
            cx0, cx1 = center_rect()
            glow_draw.rectangle([cx0 // glow_scale, gy_top, cx1 // glow_scale, gy_bot], fill=color)
        else:
            rx0, rx1, lx0, lx1 = bar_rects(i)
            glow_draw.rectangle([rx0 // glow_scale, gy_top, rx1 // glow_scale, gy_bot], fill=color)
            glow_draw.rectangle([lx0 // glow_scale, gy_top, lx1 // glow_scale, gy_bot], fill=color)

    # Blur and up-sample glow
    glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=5))
    glow_img = glow_img.resize((W, H), Image.BILINEAR)
    glow_img = ImageEnhance.Brightness(glow_img).enhance(0.45)
    img = ImageChops.add(img, glow_img)

    # --- Draw sharp bars on top ---
    WAVE_AMP   = 3      # 최대 수평 이동 픽셀
    WAVE_FREQ  = 0.07   # 수직 방향 파장 (rad/px)
    WAVE_SPEED = 0.12   # 시간 방향 속도 (rad/frame)
    SLICE      = 2      # 반사 슬라이스 두께 (px)

    # 반사 레이어 (블러 적용을 위해 별도 이미지)
    refl_img = Image.new('RGB', (W, H), (0, 0, 0))
    refl_draw = ImageDraw.Draw(refl_img)

    draw = ImageDraw.Draw(img)
    for i in range(_n_bars):
        h_val = bars[i]
        if h_val < 0.005:
            continue
        up_h = max(1, int(h_val * up_max))
        down_h = max(1, int(h_val * down_max))
        color = tuple(_bar_colors[i])
        y_top = center_y - up_h
        y_bot = center_y + down_h

        # x 좌표 미리 계산
        if i == 0:
            cx0, cx1 = center_rect()
        else:
            rx0, rx1, lx0, lx1 = bar_rects(i)

        # 위쪽 막대 (기존 동일)
        if i == 0:
            draw.rectangle([cx0, y_top, cx1, center_y], fill=color)
        else:
            draw.rectangle([rx0, y_top, rx1, center_y], fill=color)
            draw.rectangle([lx0, y_top, lx1, center_y], fill=color)

        # 아래쪽 막대 — 물 반사 효과 (페이드아웃 + 물결 왜곡) → 반사 레이어에 그림
        for y in range(center_y, y_bot, SLICE):
            depth = (y - center_y) / max(1, down_h)
            fade = (1.0 - depth) ** 1.5
            blended = tuple(
                int(bar_c * fade + bg_c * (1.0 - fade))
                for bar_c, bg_c in zip(color, _bg_color)
            )
            y1 = min(y + SLICE - 1, y_bot)
            wave_dx = int(WAVE_AMP * math.sin(y * WAVE_FREQ + frame_idx * WAVE_SPEED))
            if i == 0:
                refl_draw.rectangle([cx0 + wave_dx, y, cx1 + wave_dx, y1], fill=blended)
            else:
                refl_draw.rectangle([rx0 + wave_dx, y, rx1 + wave_dx, y1], fill=blended)
                refl_draw.rectangle([lx0 - wave_dx, y, lx1 - wave_dx, y1], fill=blended)

    # 반사 레이어 블러 후 합성 (검은 배경이므로 add로 자연스럽게 합성)
    refl_img = refl_img.filter(ImageFilter.GaussianBlur(radius=2))
    img = ImageChops.add(img, refl_img)

    # --- Center line ---
    line_color = tuple(max(0, c - 180) + 40 for c in _bg_color)
    draw.line([(margin_x, center_y), (W - margin_x, center_y)],
              fill=line_color, width=1)

    # --- Progress bar ---
    progress = frame_idx / max(1, _n_frames - 1)
    pb_x0 = margin_x
    pb_x1 = W - margin_x
    pb_y0 = H - progress_margin - progress_bar_h
    pb_y1 = pb_y0 + progress_bar_h
    # Track background
    draw.rectangle([pb_x0, pb_y0, pb_x1, pb_y1], fill=(40, 40, 40))
    # Filled portion
    fill_x1 = pb_x0 + int((pb_x1 - pb_x0) * progress)
    if fill_x1 > pb_x0:
        accent = tuple(_bar_colors[_n_bars // 2])
        draw.rectangle([pb_x0, pb_y0, fill_x1, pb_y1], fill=accent)

    # Time text centered on progress bar
    elapsed_s = frame_idx / _fps
    total_s = _duration
    elapsed_str = f"{int(elapsed_s // 60):02d}:{elapsed_s % 60:05.2f}"
    total_str   = f"{int(total_s // 60):02d}:{total_s % 60:05.2f}"
    time_text = f"{elapsed_str} / {total_str}"
    # Measure text width for centering
    bbox = _font_small.getbbox(time_text)
    tw = bbox[2] - bbox[0]
    tx = (W - tw) // 2
    ty = pb_y0 + (progress_bar_h - (bbox[3] - bbox[1])) // 2 - 1
    draw.text((tx, ty), time_text, fill=(200, 200, 200), font=_font_small)

    # --- Filename label top-left ---
    label = os.path.basename(_filename)
    lx, ly = margin_x, 12
    # Drop shadow
    draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0), font=_font)
    draw.text((lx, ly), label, fill=(180, 180, 180), font=_font)

    # --- "SPECTRUM" label top-center ---
    spec_label = 'SPECTRUM'
    bbox2 = _font_small.getbbox(spec_label)
    slx = (W - (bbox2[2] - bbox2[0])) // 2
    draw.text((slx, 14), spec_label, fill=(80, 80, 80), font=_font_small)

    return np.array(img).tobytes()


# ---------------------------------------------------------------------------
# Spectrum precomputation
# ---------------------------------------------------------------------------
def precompute_spectrum(samples, sr, fps, n_bars=64, decay=0.75):
    print("Precomputing FFT spectrum...")
    n_fft = 2048
    hop_length = max(1, sr // fps)
    n_frames = int(len(samples) / hop_length) + 1

    # STFT magnitude
    D = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=hop_length, center=True))

    # Trim to n_frames (center=True can produce 1 extra column)
    D = D[:, :n_frames]

    # Mel filterbank → mel spectrogram
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_bars, fmin=20, fmax=min(20000, sr // 2))
    mel_spec = mel_fb @ D  # (n_bars, n_frames)

    # Amplitude to dB, then normalize to [0, 1]
    mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)  # (n_bars, n_frames)
    db_min = mel_db.min()
    db_max = mel_db.max()
    db_range = db_max - db_min if db_max > db_min else 1.0
    normalized = (mel_db - db_min) / db_range  # (n_bars, n_frames) in [0, 1]

    # Transpose to (n_frames, n_bars) for per-frame access
    spec = normalized.T.astype(np.float32)  # (n_frames, n_bars)

    # EMA temporal smoothing
    smoothed = np.empty_like(spec)
    smoothed[0] = spec[0]
    for i in range(1, n_frames):
        smoothed[i] = decay * smoothed[i - 1] + (1.0 - decay) * spec[i]

    print(f"  Spectrum shape: {smoothed.shape}, n_frames={n_frames}, hop={hop_length}")
    return smoothed


# ---------------------------------------------------------------------------
# Color gradient
# ---------------------------------------------------------------------------
def make_bar_colors(n_bars, base_color_hex=None):
    """Return (n_bars, 3) uint8 array.
    Default: HSV sweep 240°→330° (blue→purple→pink).
    If base_color_hex provided: sweep ±45° around that hue."""
    if base_color_hex:
        hex_str = base_color_hex.lstrip('#')
        r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h_start = (h - 45 / 360) % 1.0
        h_end   = (h + 45 / 360) % 1.0
    else:
        h_base  = random.random()
        h_start = (h_base - 45 / 360) % 1.0
        h_end   = (h_base + 45 / 360) % 1.0

    colors = np.empty((n_bars, 3), dtype=np.uint8)
    for i in range(n_bars):
        t = i / max(1, n_bars - 1)
        # Wrap-safe interpolation
        h = h_start + t * (h_end - h_start)
        h = h % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


# ---------------------------------------------------------------------------
# Main video builder
# ---------------------------------------------------------------------------
def build_video_fast(spectrum, audio_file, output_file, fps, duration,
                     bar_colors, bg_color, W, H, n_workers=None, filename_label=None):
    n_frames, n_bars = spectrum.shape

    if filename_label is None:
        filename_label = audio_file

    # --- Put spectrum in shared memory ---
    shm = SharedMemory(create=True, size=spectrum.nbytes)
    shm_arr = np.ndarray(spectrum.shape, dtype=spectrum.dtype, buffer=shm.buf)
    shm_arr[:] = spectrum

    # --- Launch ffmpeg ---
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-r', str(fps),
        '-pix_fmt', 'rgb24',
        '-i', 'pipe:0',
        '-i', audio_file,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        output_file,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # --- Worker pool ---
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    n_workers = min(n_workers, n_frames)

    init_args = (
        shm.name, spectrum.shape, spectrum.dtype,
        bar_colors, bg_color,
        W, H, n_frames, n_bars, duration, fps, filename_label,
    )

    print(f"Rendering {n_frames} frames with {n_workers} worker(s)...")

    try:
        if n_workers > 1:
            with Pool(n_workers, initializer=_worker_init, initargs=init_args) as pool:
                for i, frame_bytes in enumerate(
                        pool.imap(render_frame, range(n_frames), chunksize=4)):
                    proc.stdin.write(frame_bytes)
                    if i % (fps * 5) == 0:
                        pct = i / n_frames * 100
                        print(f"  {pct:5.1f}%  frame {i}/{n_frames}", flush=True)
        else:
            # Single-process fallback (easier to debug)
            _worker_init(*init_args)
            for i in range(n_frames):
                proc.stdin.write(render_frame(i))
                if i % (fps * 5) == 0:
                    pct = i / n_frames * 100
                    print(f"  {pct:5.1f}%  frame {i}/{n_frames}", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
        shm.close()
        shm.unlink()

    if proc.returncode != 0:
        print(f"Error: ffmpeg exited with code {proc.returncode}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MP4 FFT spectrum visualization from audio files.'
    )
    parser.add_argument('input', help='Input audio file (wav, m4a, mp3, etc.)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output MP4 file (default: <input_name>_visualized.mp4)')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate (default: 30)')
    parser.add_argument('--width', type=int, default=1920, help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Video height (default: 1080)')
    parser.add_argument('--color', default=None,
                        help='Base bar color hex, e.g. #FF6B6B (default: blue→pink gradient)')
    parser.add_argument('--bg', default='#0A0A0A', help='Background color (default: #0A0A0A)')
    parser.add_argument('--window', type=float, default=0.05,
                        help='(ignored, kept for compatibility)')
    parser.add_argument('--bars', type=int, default=64,
                        help='Number of frequency bars (default: 64)')
    parser.add_argument('--decay', type=float, default=0.75,
                        help='EMA temporal smoothing decay (default: 0.75)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Worker processes (default: cpu_count-1)')
    parser.add_argument('--preview', action='store_true',
                        help='Render only first 15 seconds (if audio is longer)')
    return parser.parse_args()


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def load_audio_ffmpeg(path):
    """Decode audio to mono float32 PCM via ffmpeg, bypassing soundfile/audioread.

    Returns (samples, sr) with the same contract as librosa.load(sr=None, mono=True).
    """
    # Probe native sample rate
    probe = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'a:0',
         '-show_entries', 'stream=sample_rate',
         '-of', 'default=noprint_wrappers=1:nokey=1',
         path],
        capture_output=True, text=True, check=True,
    )
    sr = int(probe.stdout.strip())

    # Decode to raw f32le mono PCM
    result = subprocess.run(
        ['ffmpeg', '-v', 'error',
         '-i', path,
         '-f', 'f32le', '-ac', '1', '-ar', str(sr),
         'pipe:1'],
        capture_output=True, check=True,
    )
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples, sr


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        suffix = '_preview' if args.preview else '_visualized'
        args.output = f"{base}{suffix}.mp4"

    # Load audio
    print(f"Loading audio: {args.input}")
    samples, sr = load_audio_ffmpeg(args.input)
    duration = len(samples) / sr
    print(f"  Sample rate: {sr} Hz, Duration: {duration:.2f}s")

    # Preview mode: trim to 15 seconds
    PREVIEW_SECS = 15
    if args.preview and duration > PREVIEW_SECS:
        samples = samples[:int(PREVIEW_SECS * sr)]
        duration = PREVIEW_SECS
        print(f"  Preview mode: trimming to {PREVIEW_SECS}s")

    # Precompute spectrum
    spectrum = precompute_spectrum(samples, sr, args.fps, n_bars=args.bars, decay=args.decay)

    # Colors
    bar_colors = make_bar_colors(args.bars, base_color_hex=args.color)
    bg_color = hex_to_rgb(args.bg)

    # Render + encode
    print(f"Rendering video: {args.output}")
    build_video_fast(
        spectrum=spectrum,
        audio_file=args.input,
        output_file=args.output,
        fps=args.fps,
        duration=duration,
        bar_colors=bar_colors,
        bg_color=bg_color,
        W=args.width,
        H=args.height,
        n_workers=args.workers,
        filename_label=args.input,
    )
    print(f"Done: {args.output}")


if __name__ == '__main__':
    main()
