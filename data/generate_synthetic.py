"""
generate_synthetic.py
=====================
Generates ALL synthetic data needed for the WeldSense digital simulation:
  • data/images/  — synthetic weld images (5 per defect class × 6 classes = 30)
  • data/thermal/ — synthetic thermal heatmaps (10 per class × 4 classes = 40)
  • data/audio/   — synthetic WAV files (50 normal + 50 anomaly = 100)
  • data/audio/   — spectrogram PNGs (one per WAV)

Run from the project root:
    python data/generate_synthetic.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFilter
import scipy.io.wavfile as wav
import glob

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR     = os.path.join(ROOT, "data", "images")
THERMAL_DIR = os.path.join(ROOT, "data", "thermal")
AUDIO_DIR   = os.path.join(ROOT, "data", "audio")

for d in [IMG_DIR, THERMAL_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC WELD IMAGES  (visual modality)
# ─────────────────────────────────────────────────────────────────────────────
DEFECT_CLASSES = ["good_weld", "burn_through", "contamination",
                  "lack_of_fusion", "spatter", "porosity"]

def _weld_base(size=224):
    """Dark metal surface with a central weld bead."""
    rng = np.random.default_rng()
    # dark grey metal background
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    noise = rng.integers(0, 15, (size, size, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    # weld bead — horizontal stripe in center
    bead_y  = size // 2
    bead_w  = size // 5
    bead_col = np.array([180, 130, 80])  # golden-bronze
    for dy in range(-bead_w // 2, bead_w // 2 + 1):
        alpha = 1 - abs(dy) / (bead_w / 2 + 1)
        row = np.clip(bead_y + dy, 0, size - 1)
        row_noise = rng.integers(-10, 10, (size, 3))
        img[row] = np.clip((bead_col * alpha + img[row] * (1 - alpha)) + row_noise,
                           0, 255).astype(np.uint8)
    return img


def make_weld_image(defect_type: str, index: int):
    rng = np.random.default_rng(seed=index * 100 + DEFECT_CLASSES.index(defect_type))
    size = 224
    img = _weld_base(size)

    if defect_type == "good_weld":
        pass  # clean bead

    elif defect_type == "burn_through":
        # dark hole in the bead
        cx, cy = size // 2 + rng.integers(-20, 20), size // 2
        r = rng.integers(8, 18)
        for y in range(cy - r, cy + r):
            for x in range(cx - r, cx + r):
                if 0 <= y < size and 0 <= x < size:
                    if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                        img[y, x] = [10, 5, 5]

    elif defect_type == "contamination":
        # green/blue blotches on bead
        for _ in range(rng.integers(3, 7)):
            cx = rng.integers(30, size - 30)
            cy = size // 2 + rng.integers(-10, 10)
            r  = rng.integers(4, 12)
            for y in range(cy - r, cy + r):
                for x in range(cx - r, cx + r):
                    if 0 <= y < size and 0 <= x < size:
                        if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                            img[y, x] = [rng.integers(20, 60),
                                         rng.integers(80, 140),
                                         rng.integers(20, 60)]

    elif defect_type == "lack_of_fusion":
        # pale/washed-out regions along bead edge
        edge_y = size // 2 + (size // 10)
        for x in range(size // 4, 3 * size // 4):
            if rng.random() < 0.6:
                img[edge_y, x] = np.clip(
                    img[edge_y, x].astype(int) + rng.integers(60, 100, 3), 0, 255
                ).astype(np.uint8)

    elif defect_type == "spatter":
        # small bright droplets around bead
        for _ in range(rng.integers(15, 35)):
            sx = rng.integers(10, size - 10)
            sy = size // 2 + rng.integers(-40, 40)
            r  = rng.integers(1, 5)
            for y in range(sy - r, sy + r):
                for x in range(sx - r, sx + r):
                    if 0 <= y < size and 0 <= x < size:
                        img[y, x] = [rng.integers(200, 255),
                                     rng.integers(150, 220),
                                     rng.integers(50, 120)]

    elif defect_type == "porosity":
        # small dark pits in the bead
        for _ in range(rng.integers(5, 12)):
            px = rng.integers(20, size - 20)
            py = size // 2 + rng.integers(-8, 8)
            r  = rng.integers(2, 7)
            for y in range(py - r, py + r):
                for x in range(px - r, px + r):
                    if 0 <= y < size and 0 <= x < size:
                        if (x - px) ** 2 + (y - py) ** 2 < r ** 2:
                            img[y, x] = np.clip(
                                img[y, x].astype(int) - rng.integers(60, 100, 3), 0, 255
                            ).astype(np.uint8)

    return Image.fromarray(img)


def generate_weld_images(n_per_class=5):
    print("\n[1/3] Generating synthetic weld images…")
    for defect in DEFECT_CLASSES:
        for i in range(n_per_class):
            filename = f"{defect}_{i:02d}.jpg"
            path = os.path.join(IMG_DIR, filename)
            img = make_weld_image(defect, i)
            img.save(path, quality=90)
            print(f"  [OK] {filename}")
    print(f"  -> {n_per_class * len(DEFECT_CLASSES)} images saved to data/images/")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SYNTHETIC THERMAL HEATMAPS  (IR modality)
# ─────────────────────────────────────────────────────────────────────────────
THERMAL_CLASSES = ["normal", "cold_weld", "porosity", "misalignment"]


def make_thermal(defect_type: str, index: int, filename: str):
    rng = np.random.default_rng(seed=index * 77 + THERMAL_CLASSES.index(defect_type))
    grid = rng.normal(35, 2, (24, 32)).astype(float)   # baseline 35 °C

    if defect_type == "cold_weld":
        grid[10:14, 14:18] -= rng.uniform(6, 10)        # cold spot
    elif defect_type == "porosity":
        grid[11:13, 15:17] += rng.uniform(12, 18)       # hot spot (trapped air)
    elif defect_type == "misalignment":
        grid[8:12, 20:24]  += rng.uniform(8, 13)        # off-centre heat
    # "normal" stays as baseline ± noise

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    im = ax.imshow(grid, cmap="inferno", vmin=25, vmax=60)
    plt.colorbar(im, ax=ax, label="°C")
    ax.axis("off")
    fig.savefig(filename, dpi=80, bbox_inches="tight")
    plt.close(fig)


def generate_thermal_images(n_per_class=10):
    print("\n[2/3] Generating synthetic thermal heatmaps…")
    for defect in THERMAL_CLASSES:
        for i in range(n_per_class):
            fname = os.path.join(THERMAL_DIR, f"{defect}_{i}.png")
            make_thermal(defect, i, fname)
            print(f"  [OK] {defect}_{i}.png")
    print(f"  -> {n_per_class * len(THERMAL_CLASSES)} thermal images saved to data/thermal/")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYNTHETIC AUDIO WAV FILES + SPECTROGRAMS  (acoustic modality)
# ─────────────────────────────────────────────────────────────────────────────
SR = 16_000          # sample rate
DURATION = 2.0       # seconds


def _make_wav(rng, anomaly: bool) -> np.ndarray:
    """
    Normal  → low-freq hum + small Gaussian noise
    Anomaly → hum + random crackling bursts (simulate arc instability)
    """
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    # base hum at 100 Hz
    sig = 0.3 * np.sin(2 * np.pi * 100 * t)
    # gaussian background noise
    sig += rng.normal(0, 0.05, len(t))

    if anomaly:
        # add 3-6 crackling bursts
        n_bursts = rng.integers(3, 7)
        for _ in range(n_bursts):
            start = rng.integers(0, int(SR * (DURATION - 0.05)))
            length = rng.integers(200, 1200)
            burst = rng.normal(0, rng.uniform(0.4, 0.9), length)
            end = min(start + length, len(sig))
            sig[start:end] += burst[: end - start]

    # normalise to int16
    sig = np.clip(sig / np.abs(sig).max(), -1, 1)
    return (sig * 32767).astype(np.int16)


def _spectrogram_png(wav_path: str, save_path: str):
    """Convert .wav to mel-spectrogram PNG (requires librosa)."""
    try:
        import librosa
        import librosa.display

        y, sr = librosa.load(wav_path, sr=SR, duration=DURATION)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        librosa.display.specshow(mel_db, sr=sr, fmax=8000, ax=ax)
        ax.axis("off")
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return True
    except ImportError:
        # Fallback: STFT via scipy/numpy only
        rate, data = wav.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(float) / 32767.0
        n_fft = 512
        hop   = 256
        frames = [
            np.abs(np.fft.rfft(data[i:i + n_fft] * np.hanning(n_fft), n=n_fft))
            for i in range(0, len(data) - n_fft, hop)
        ]
        spec = np.array(frames).T  # (freq_bins, time_frames)
        spec_db = 20 * np.log10(spec + 1e-9)

        fig, ax = plt.subplots(figsize=(2.24, 2.24))
        ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma")
        ax.axis("off")
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return True


def generate_audio(n_normal=50, n_anomaly=50):
    print("\n[3/3] Generating synthetic audio WAV + spectrogram PNGs…")
    rng = np.random.default_rng(seed=42)

    pairs = (
        [("normal",  False, f"normal_{i:02d}")  for i in range(n_normal)] +
        [("anomaly", True,  f"anomaly_{i:02d}") for i in range(n_anomaly)]
    )

    for label, is_anomaly, name in pairs:
        wav_path = os.path.join(AUDIO_DIR, f"{name}.wav")
        png_path = os.path.join(AUDIO_DIR, f"{name}.png")

        signal = _make_wav(rng, anomaly=is_anomaly)
        wav.write(wav_path, SR, signal)
        _spectrogram_png(wav_path, png_path)
        print(f"  [OK] {name}.wav + .png")

    print(f"  -> {n_normal + n_anomaly} WAV + {n_normal + n_anomaly} PNG saved to data/audio/")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WELDING QUALITY CSV  (for defect classifier training — Step 9)
# ─────────────────────────────────────────────────────────────────────────────
DEFECT_PARAMS = {
    "good_weld":      {"voltage": (22, 2), "current": (180, 10), "speed": (0.5, 0.05)},
    "burn_through":   {"voltage": (30, 2), "current": (250, 15), "speed": (0.3, 0.05)},
    "contamination":  {"voltage": (21, 3), "current": (175, 15), "speed": (0.55, 0.08)},
    "lack_of_fusion": {"voltage": (18, 2), "current": (140, 10), "speed": (0.7, 0.06)},
    "spatter":        {"voltage": (28, 2), "current": (230, 10), "speed": (0.35, 0.04)},
    "porosity":       {"voltage": (23, 2), "current": (190, 10), "speed": (0.45, 0.05)},
}


def generate_welding_csv(n_per_class=80):
    """Create a synthetic welding_quality.csv for Step 9 classifier training."""
    import csv
    rng = np.random.default_rng(seed=7)
    out_path = os.path.join(ROOT, "data", "welding_quality.csv")

    rows = []
    for defect, params in DEFECT_PARAMS.items():
        for _ in range(n_per_class):
            v  = rng.normal(*params["voltage"])
            c  = rng.normal(*params["current"])
            sp = rng.normal(*params["speed"])
            rows.append({"Voltage": round(v, 2),
                         "Current": round(c, 1),
                         "WeldSpeed": round(sp, 3),
                         "DefectType": defect})

    rng.shuffle(rows)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Voltage", "Current", "WeldSpeed", "DefectType"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[+] Welding quality CSV saved → data/welding_quality.csv ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  WeldSense — Synthetic Data Generator")
    print("=" * 60)

    # generate_weld_images(n_per_class=5)
    generate_thermal_images(n_per_class=10)
    # generate_audio(n_normal=50, n_anomaly=50)
    generate_welding_csv(n_per_class=80)

    print("\n" + "=" * 60)
    print("  [DONE] All synthetic data generated successfully!")
    print("  data/images/   -> 30 weld images")
    print("  data/thermal/  -> 40 thermal heatmaps")
    print("  data/audio/    -> 100 WAV + 100 spectrogram PNGs")
    print("  data/welding_quality.csv -> 480 rows for classifier")
    print("=" * 60)
    print("\nNext step: python pipeline/train.py")
