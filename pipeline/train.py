"""
train.py
========
One-shot training script for WeldSense AI models.

Run from the project root after generating synthetic data:
    python pipeline/train.py

What it does
------------
1. Trains AnomalyDetector      (IsolationForest on normal spectrogram PNGs)
2. Trains DefectClassifier     (RandomForest on welding_quality.csv)
3. Trains ImageDefectClassifier (RandomForest on visual features from data/images/)
4. Saves all models to models/

After this script completes, the dashboard and inspect_cell() are fully
operational without any additional setup.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.models import AnomalyDetector, DefectClassifier, ImageDefectClassifier
from pipeline.database import init_db


def main():
    print("=" * 60)
    print("  WeldSense — Model Training")
    print("=" * 60)

    models_dir   = os.path.join(ROOT, "models")
    audio_dir    = os.path.join(ROOT, "data", "audio")
    images_dir   = os.path.join(ROOT, "data", "images")
    csv_path     = os.path.join(ROOT, "data", "welding_quality.csv")
    ad_pkl       = os.path.join(models_dir, "anomaly_detector.pkl")
    clf_pkl      = os.path.join(models_dir, "defect_classifier.pkl")
    enc_pkl      = os.path.join(models_dir, "label_encoder.pkl")
    img_clf_pkl  = os.path.join(models_dir, "image_classifier.pkl")
    img_enc_pkl  = os.path.join(models_dir, "image_label_encoder.pkl")

    os.makedirs(models_dir, exist_ok=True)

    # ── pre-flight checks ────────────────────────────────────────────────────
    png_count = len([f for f in os.listdir(audio_dir) if f.endswith(".png")])
    if png_count < 10:
        print(f"\n[!] Only {png_count} spectrogram PNGs found in data/audio/")
        print("    Run first:  python data/generate_synthetic.py")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"\n[!] welding_quality.csv not found at {csv_path}")
        print("    Run first:  python data/generate_synthetic.py")
        sys.exit(1)

    jpg_count = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    if jpg_count < 4:
        print(f"\n[!] Only {jpg_count} images found in data/images/")
        print("    Run first:  python data/generate_synthetic.py")
        sys.exit(1)

    # ── Step 8: Anomaly Detector ─────────────────────────────────────────────
    print("\n── Anomaly Detector (IsolationForest on spectrograms) ──")
    anomaly_det = AnomalyDetector(contamination=0.1, random_state=42)
    eval_ad     = anomaly_det.train(audio_dir=audio_dir, verbose=True)
    anomaly_det.save(ad_pkl)

    if "separation" in eval_ad:
        sep = eval_ad["separation"]
        if sep > 0.05:
            print(f"  [OK] Good separation (delta={sep:.4f}) -- model is discriminating")
        else:
            print(f"  [!] Low separation (delta={sep:.4f}) -- consider more training data")

    # ── Step 9: Defect Classifier (param-based) ──────────────────────────────
    print("\n── Defect Classifier (RandomForest on weld parameters) ──")
    defect_clf = DefectClassifier(n_estimators=100, random_state=42)
    eval_clf   = defect_clf.train(csv_path=csv_path, verbose=True)
    defect_clf.save(clf_pkl, enc_pkl)

    acc = eval_clf["accuracy"]
    if acc >= 0.85:
        print(f"  [OK] Accuracy: {acc:.2%} -- excellent")
    elif acc >= 0.70:
        print(f"  [OK] Accuracy: {acc:.2%} -- acceptable")
    else:
        print(f"  [!] Accuracy: {acc:.2%} -- low (more data may help)")

    # ── Step 8b: Image Defect Classifier (visual modality) ───────────────────
    print("\n── Image Defect Classifier (RandomForest on visual features) ──")
    img_clf = ImageDefectClassifier(n_estimators=200, random_state=42)
    try:
        eval_img = img_clf.train(images_dir=images_dir, verbose=True)
        img_clf.save(img_clf_pkl, img_enc_pkl)

        img_acc = eval_img["accuracy"]
        if img_acc >= 0.80:
            print(f"  [OK] Image accuracy: {img_acc:.2%} -- excellent")
        elif img_acc >= 0.60:
            print(f"  [OK] Image accuracy: {img_acc:.2%} -- acceptable")
        else:
            print(f"  [!] Image accuracy: {img_acc:.2%} -- low")
            print("       Consider adding more labelled images per defect class.")
    except RuntimeError as e:
        print(f"  [!] Image classifier skipped: {e}")
        print("      Parameter-based classifier will be used as fallback.")

    # ── Initialise database ──────────────────────────────────────────────────
    print("\n── Initialising SQLite database ──")
    init_db()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  [DONE] Training complete!")
    print(f"  models/anomaly_detector.pkl    -- anomaly score model")
    print(f"  models/defect_classifier.pkl   -- param-based defect classifier")
    print(f"  models/image_classifier.pkl    -- visual defect classifier (Layer 1)")
    print(f"  models/label_encoder.pkl       -- param class mapping")
    print(f"  models/image_label_encoder.pkl -- image class mapping")
    print(f"  data/inspections.db            -- SQLite database")
    print("=" * 60)
    print("\nStart the dashboard:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
