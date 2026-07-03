"""
run.py — WeldSense one-click launcher
=======================================
Run from the project root:
    python run.py

Workflow
--------
1. Install / verify Python dependencies
2. Generate all synthetic data  (if not already present)
3. Train both AI models          (if models not already saved)
4. Launch the Streamlit dashboard

Skip flags (for faster re-runs):
    python run.py --skip-data     skip data generation
    python run.py --skip-train    skip model training
    python run.py --skip-data --skip-train  go straight to dashboard
"""

import os
import sys
import subprocess
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list, label: str):
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] '{' '.join(cmd)}' failed with code {result.returncode}")
        sys.exit(result.returncode)


def data_ready() -> bool:
    img_dir     = os.path.join(ROOT, "data", "images")
    thermal_dir = os.path.join(ROOT, "data", "thermal")
    audio_dir   = os.path.join(ROOT, "data", "audio")
    return (
        len([f for f in os.listdir(img_dir)     if f.endswith(".jpg")]) >= 20
        and len([f for f in os.listdir(thermal_dir) if f.endswith(".png")]) >= 30
        and len([f for f in os.listdir(audio_dir)   if f.endswith(".wav")]) >= 50
    )


def models_ready() -> bool:
    models_dir = os.path.join(ROOT, "models")
    return (
        os.path.exists(os.path.join(models_dir, "anomaly_detector.pkl"))
        and os.path.exists(os.path.join(models_dir, "defect_classifier.pkl"))
        and os.path.exists(os.path.join(models_dir, "label_encoder.pkl"))
    )


def main():
    parser = argparse.ArgumentParser(description="WeldSense launcher")
    parser.add_argument("--skip-data",  action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    print("""
  ██████╗  █████╗ ████████╗████████╗██╗     ███████╗███████╗██████╗  ██████╗ ███████╗
  ██╔══██╗██╔══██╗╚══██╔══╝╚══██╔══╝██║     ██╔════╝██╔════╝██╔══██╗██╔════╝ ██╔════╝
  ██████╔╝███████║   ██║      ██║   ██║     █████╗  █████╗  ██║  ██║██║  ███╗█████╗
  ██╔══██╗██╔══██║   ██║      ██║   ██║     ██╔══╝  ██╔══╝  ██║  ██║██║   ██║██╔══╝
  ██████╔╝██║  ██║   ██║      ██║   ███████╗███████╗███████╗██████╔╝╚██████╔╝███████╗
  ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚══════╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝
  EV Battery Cell Weld Inspection System — Digital Prototype
    """)

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    if args.skip_data:
        print("[skip] Data generation skipped.")
    elif data_ready():
        print("[✓] Synthetic data already present — skipping generation.")
    else:
        run([py, "data/generate_synthetic.py"], "Generating Synthetic Data")

    # ── Step 2: Train models ──────────────────────────────────────────────────
    if args.skip_train:
        print("[skip] Model training skipped.")
    elif models_ready():
        print("[✓] Trained models already present — skipping training.")
    else:
        run([py, "pipeline/train.py"], "Training AI Models")

    # ── Step 3: Launch dashboard ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Launching WeldSense Dashboard…")
    print("  Open:  http://localhost:8501")
    print("="*55 + "\n")
    subprocess.run(
        [py, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
