"""
sensors.py
==========
Sensor reading functions for the WeldSense inspection pipeline.

  read_weld_image(filepath)       → (224,224,3) normalised array + original BGR
  extract_image_features(filepath) → (28,) lighting-invariant feature vector
  read_thermal(filepath)          → normalised array + {"peak", "mean", "std"} dict
  audio_to_spectrogram(path)      → mel-spectrogram ndarray (128, time_steps)

All functions are pure (no side-effects) and work with either real sensor
data or the synthetic files produced by data/generate_synthetic.py.
"""

import os
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CAMERA  — visual weld image
# ─────────────────────────────────────────────────────────────────────────────

def read_weld_image(filepath: str):
    """
    Read a weld image and return a model-ready array plus the display version.

    Parameters
    ----------
    filepath : str  path to .jpg / .png image

    Returns
    -------
    img_normalized : np.ndarray  shape (224, 224, 3), float32, values 0-1
    img_display    : np.ndarray  shape (H, W, 3), uint8, RGB colour order
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filepath}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to 224×224 (standard input for most vision models)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # normalise pixel values to 0-1 float32
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized, img_rgb


def extract_image_features(filepath: str) -> np.ndarray:
    """
    Extract a 28-dimensional lighting-invariant feature vector from a weld image.

    All features are designed to be robust to:
    - Different camera exposures / brightness levels
    - Varying distances and resolutions
    - Different background surfaces

    Key approach: normalise each image to its own mean first, then extract
    relative structural features (blob counts, edge ratios, etc.) so that
    a dark photo and a bright photo of the same defect type produce similar
    feature vectors.

    Parameters
    ----------
    filepath : str  path to .jpg / .png weld image

    Returns
    -------
    features : np.ndarray  shape (28,), float32
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filepath}")

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # ── Normalise to own mean to remove lighting bias ─────────────────────────
    gray_mean_val = img_gray.mean()
    gray_norm     = img_gray / (gray_mean_val + 1e-5)   # relative brightness

    # Bead centre strip (rows 80–144, 64px tall)
    strip_gray = img_gray[80:144, :]
    strip_norm = gray_norm[80:144, :]

    # ── 1. Relative colour fractions (3) — lighting-invariant ─────────────────
    r_mean, g_mean, b_mean = (
        img_rgb[:, :, 0].mean(),
        img_rgb[:, :, 1].mean(),
        img_rgb[:, :, 2].mean(),
    )
    total_intensity = r_mean + g_mean + b_mean + 1e-5
    r_frac = float(r_mean / total_intensity)
    g_frac = float(g_mean / total_intensity)   # contamination → higher
    b_frac = float(b_mean / total_intensity)

    # ── 2. Colour standard deviations (3) ─────────────────────────────────────
    r_std = float(img_rgb[:, :, 0].std())
    g_std = float(img_rgb[:, :, 1].std())
    b_std = float(img_rgb[:, :, 2].std())

    # ── 3. HSV saturation + hue in bead strip (4) ─────────────────────────────
    sat_full  = img_hsv[:, :, 1] / 255.0
    hue_strip = img_hsv[80:144, :, 0] / 180.0
    sat_mean  = float(sat_full.mean())
    sat_std   = float(sat_full.std())
    hue_mean  = float(hue_strip.mean())
    hue_std   = float(hue_strip.std())

    # ── 4. Edge density global + strip + ratio (3) ────────────────────────────
    img_u8   = (img_gray * 255).astype(np.uint8)
    strip_u8 = (strip_gray * 255).astype(np.uint8)
    edge_glob = float(cv2.Canny(img_u8, 40, 120).mean()) / 255.0
    edge_str  = float(cv2.Canny(strip_u8, 40, 120).mean()) / 255.0
    edge_ratio = edge_str / (edge_glob + 1e-5)

    # ── 5. BURN-THROUGH: large dark blobs relative to image mean (3) ──────────
    dark_mask_strip = (gray_norm[80:144, :] < 0.40).astype(np.uint8)
    n_dark, _, d_stats, _ = cv2.connectedComponentsWithStats(dark_mask_strip)
    if n_dark > 1:
        max_dark_area = float(d_stats[1:, cv2.CC_STAT_AREA].max()) / (64 * 224)
        large_dark    = int((d_stats[1:, cv2.CC_STAT_AREA] > 100).sum())
    else:
        max_dark_area, large_dark = 0.0, 0
    dark_ratio = float(dark_mask_strip.sum()) / dark_mask_strip.size

    # ── 6. SPATTER: small bright blobs relative to image mean (3) ────────────
    bright_mask = (gray_norm > 1.60).astype(np.uint8)
    n_brt, _, b_stats, _ = cv2.connectedComponentsWithStats(bright_mask)
    if n_brt > 1:
        small_bright = int((b_stats[1:, cv2.CC_STAT_AREA] < 30).sum())
        total_bright = int(b_stats[1:, cv2.CC_STAT_AREA].sum())
    else:
        small_bright, total_bright = 0, 0
    bright_ratio = float(bright_mask.sum()) / bright_mask.size

    # ── 7. POROSITY: tiny dark pits in bead strip (2) ────────────────────────
    pore_mask = (strip_norm < 0.50).astype(np.uint8)
    n_pore, _, p_stats, _ = cv2.connectedComponentsWithStats(pore_mask)
    if n_pore > 1:
        tiny_pores   = int((p_stats[1:, cv2.CC_STAT_AREA] < 20).sum())
        medium_pores = int(((p_stats[1:, cv2.CC_STAT_AREA] >= 20) &
                            (p_stats[1:, cv2.CC_STAT_AREA] < 100)).sum())
    else:
        tiny_pores, medium_pores = 0, 0

    # ── 8. LACK OF FUSION: bead edge asymmetry (2) ───────────────────────────
    top_norm    = float(strip_norm[:10, :].mean())
    bottom_norm = float(strip_norm[-10:, :].mean())
    edge_asym   = abs(top_norm - bottom_norm)
    lof_streak  = float(strip_norm[:10, :].std())

    # ── 9. Texture: scale-invariant patch variance in strip (3) ──────────────
    block_vars = []
    for i in range(0, 64, 8):
        for j in range(0, 224, 16):
            block_vars.append(strip_gray[i:i+8, j:j+16].var())
    patch_var_mean = float(np.mean(block_vars))
    patch_var_std  = float(np.std(block_vars))
    patch_var_cv   = patch_var_std / (patch_var_mean + 1e-8)  # scale-invariant

    # ── 10. Green excess (1) + strip relative std (1) ────────────────────────
    green_rel   = float(g_frac - (r_frac + b_frac) / 2.0)
    strip_rstd  = float(strip_norm.std())

    features = np.array([
        # Relative colour (3)
        r_frac, g_frac, b_frac,
        # Colour std (3)
        r_std, g_std, b_std,
        # HSV (4)
        sat_mean, sat_std, hue_mean, hue_std,
        # Edge (3)
        edge_glob, edge_str, edge_ratio,
        # Burn-through (3)
        max_dark_area, float(large_dark), dark_ratio,
        # Spatter (3)
        float(small_bright), float(total_bright) / (224 * 224), bright_ratio,
        # Porosity (2)
        float(tiny_pores), float(medium_pores),
        # Lack of fusion (2)
        edge_asym, lof_streak,
        # Texture (3)
        patch_var_mean, patch_var_std, patch_var_cv,
        # Misc (2)
        green_rel, strip_rstd,
    ], dtype=np.float32)  # total = 28

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IR SENSOR  — thermal heatmap
# ─────────────────────────────────────────────────────────────────────────────

def read_thermal(filepath: str):
    """
    Read a thermal heatmap PNG and extract three scalar features.

    Parameters
    ----------
    filepath : str  path to thermal PNG

    Returns
    -------
    thermal_norm : np.ndarray  shape (H, W), float32, values 0-1
    features     : dict        {"peak", "mean", "std"}
                    peak → highest temperature proxy  (higher = hotter spot)
                    mean → average temperature        (higher = generally hot)
                    std  → temperature unevenness     (higher = suspect weld)
    """
    thermal = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if thermal is None:
        raise FileNotFoundError(f"Thermal image not found: {filepath}")

    thermal_norm = thermal.astype(np.float32) / 255.0

    peak_temp = float(np.max(thermal_norm))
    mean_temp = float(np.mean(thermal_norm))
    std_temp  = float(np.std(thermal_norm))

    features = {
        "peak": round(peak_temp, 3),
        "mean": round(mean_temp, 3),
        "std":  round(std_temp,  3),
    }

    return thermal_norm, features


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MICROPHONE  — acoustic spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def audio_to_spectrogram(wav_path: str, save_path: str = None):
    """
    Convert a WAV file to a mel-spectrogram 2-D array.

    Uses librosa if available; falls back to numpy STFT otherwise.

    Parameters
    ----------
    wav_path  : str   input .wav file
    save_path : str | None   if given, save a 224×224 PNG of the spectrogram

    Returns
    -------
    mel_db : np.ndarray  shape (128, time_steps), values in dB  (librosa path)
           OR STFT magnitude array (fallback path)
    """
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, sr = librosa.load(wav_path, sr=16_000, duration=2.0)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=128,
            fmax=8_000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if save_path:
            fig, ax = plt.subplots(figsize=(2.24, 2.24))
            librosa.display.specshow(mel_db, sr=sr, fmax=8_000, ax=ax)
            ax.axis("off")
            fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return mel_db

    except ImportError:
        # ── Fallback: scipy + numpy STFT ──────────────────────────────────
        import scipy.io.wavfile as wav_io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rate, data = wav_io.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32) / 32767.0

        n_fft = 512
        hop   = 256
        frames = [
            np.abs(np.fft.rfft(data[i:i + n_fft] * np.hanning(n_fft), n=n_fft))
            for i in range(0, max(1, len(data) - n_fft), hop)
        ]
        spec = np.array(frames).T  # (freq_bins, time_frames)
        spec_db = 20 * np.log10(spec + 1e-9)

        if save_path:
            fig, ax = plt.subplots(figsize=(2.24, 2.24))
            ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma")
            ax.axis("off")
            fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return spec_db


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR     = os.path.join(ROOT, "data", "images")
    THERMAL_DIR = os.path.join(ROOT, "data", "thermal")
    AUDIO_DIR   = os.path.join(ROOT, "data", "audio")

    # ── visual ──
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if img_files:
        arr, disp = read_weld_image(os.path.join(IMG_DIR, img_files[0]))
        feat = extract_image_features(os.path.join(IMG_DIR, img_files[0]))
        print(f"[visual]  shape={arr.shape}, features shape={feat.shape}")
    else:
        print("[visual]  No images found.")

    # ── thermal ──
    th_files = [f for f in os.listdir(THERMAL_DIR) if f.endswith(".png")]
    if th_files:
        th, feat = read_thermal(os.path.join(THERMAL_DIR, th_files[0]))
        print(f"[thermal] shape={th.shape}, features={feat}")
    else:
        print("[thermal] No thermal images found.")

    # ── audio ──
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    if wav_files:
        spec = audio_to_spectrogram(os.path.join(AUDIO_DIR, wav_files[0]))
        print(f"[audio]   spectrogram shape={spec.shape}")
    else:
        print("[audio]   No WAV files found.")
