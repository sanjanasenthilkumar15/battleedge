"""
pipeline_core.py
================
Core WeldSense inspection pipeline — Steps 5-11.

Public API
----------
inspect_cell(cell_id, image_path, audio_path, thermal_path,
             voltage, current, weld_speed)
    → dict with all inspection fields

The function loads pre-trained models on first call (lazy singleton),
so repeated calls in a Streamlit session are fast.
"""

import os
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lazy-loaded model singletons — avoid reloading on every call
_anomaly_detector  = None
_defect_classifier = None
_image_classifier  = None


def _load_models():
    """Load or return cached model objects."""
    global _anomaly_detector, _defect_classifier, _image_classifier

    if _anomaly_detector is None or _defect_classifier is None:
        from pipeline.models import AnomalyDetector, DefectClassifier
        models_dir = os.path.join(ROOT, "models")

        ad_path  = os.path.join(models_dir, "anomaly_detector.pkl")
        clf_path = os.path.join(models_dir, "defect_classifier.pkl")
        enc_path = os.path.join(models_dir, "label_encoder.pkl")

        if not os.path.exists(ad_path) or not os.path.exists(clf_path):
            raise FileNotFoundError(
                "Trained models not found.\n"
                "Run:  python pipeline/train.py"
            )

        _anomaly_detector  = AnomalyDetector.load(ad_path)
        _defect_classifier = DefectClassifier.load(clf_path, enc_path)

    # Load image classifier if available (optional — graceful fallback)
    if _image_classifier is None:
        from pipeline.models import ImageDefectClassifier
        img_clf_path = os.path.join(ROOT, "models", "image_classifier.pkl")
        img_enc_path = os.path.join(ROOT, "models", "image_label_encoder.pkl")
        if os.path.exists(img_clf_path):
            try:
                _image_classifier = ImageDefectClassifier.load(img_clf_path, img_enc_path)
            except Exception:
                _image_classifier = None  # silently fall back to param-only

    return _anomaly_detector, _defect_classifier


# ─────────────────────────────────────────────────────────────────────────────
# Default simulated weld parameters per defect type
# Used when the dashboard doesn't supply real sensor readings
# ─────────────────────────────────────────────────────────────────────────────
_SIM_PARAMS = {
    "good_weld":      (22.0, 180.0, 0.50),
    "normal":         (22.0, 180.0, 0.50),
    "burn_through":   (30.0, 250.0, 0.30),
    "contamination":  (21.0, 175.0, 0.55),
    "lack_of_fusion": (18.0, 140.0, 0.70),
    "spatter":        (28.0, 230.0, 0.35),
    "porosity":       (23.0, 190.0, 0.45),
    "cold_weld":      (20.0, 155.0, 0.60),
    "misalignment":   (24.0, 200.0, 0.40),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function  — Step 11
# ─────────────────────────────────────────────────────────────────────────────

def inspect_cell(
    cell_id:      str,
    image_path:   str,
    audio_path:   str,
    thermal_path: str,
    voltage:      float = None,
    current:      float = None,
    weld_speed:   float = None,
    sim_defect:   str   = None,
) -> dict:
    """
    Run the full WeldSense multi-modal inspection pipeline.

    Parameters
    ----------
    cell_id      : str   unique cell identifier, e.g. "CELL-001"
    image_path   : str   path to weld image (.jpg/.png)
    audio_path   : str   path to audio WAV file
    thermal_path : str   path to thermal PNG
    voltage      : float optional real process parameter (V)
    current      : float optional real process parameter (A)
    weld_speed   : float optional real process parameter (m/s)
    sim_defect   : str   hint used to look up simulated weld params
                         when voltage/current/weld_speed are not supplied

    Returns
    -------
    dict
        cell_id        : str
        anomaly_score  : float   (IsolationForest decision value)
        defect_type    : str
        defect_proba   : dict    {class: probability}
        thermal        : dict    {peak, mean, std}
        risk_score     : int     0-100
        decision       : "PASS" | "MONITOR" | "REJECT"
        latency_ms     : int
        image_path     : str
        audio_path     : str
        thermal_path   : str
    """
    t0 = time.time()

    # ── Step 1: Load sensors ─────────────────────────────────────────────────
    from pipeline.sensors import read_weld_image, read_thermal, audio_to_spectrogram

    img_array, _        = read_weld_image(image_path)
    spectrogram         = audio_to_spectrogram(audio_path)
    thermal_img, t_feat = read_thermal(thermal_path)

    # ── Step 2: Load models ──────────────────────────────────────────────────
    anomaly_det, defect_clf = _load_models()

    # ── Step 3: Anomaly score (acoustic) ────────────────────────────────────
    anomaly_score = anomaly_det.score(spectrogram)

    # ── Step 4a: Visual defect prediction (image-driven — Layer 1) ───────────
    from pipeline.sensors import extract_image_features
    try:
        img_features = extract_image_features(image_path)
        visual_pred  = _image_classifier.predict(img_features) if _image_classifier else None
        visual_proba = _image_classifier.predict_proba(img_features) if _image_classifier else {}
    except Exception:
        visual_pred  = None
        visual_proba = {}

    # ── Step 4b: Param-based defect prediction (fallback / secondary) ────────
    if voltage is None or current is None or weld_speed is None:
        hint = (sim_defect or "normal").lower()
        voltage, current, weld_speed = _SIM_PARAMS.get(hint, _SIM_PARAMS["normal"])

    param_pred  = defect_clf.predict(voltage, current, weld_speed)
    param_proba = defect_clf.predict_proba(voltage, current, weld_speed)

    # ── Step 4c: Fuse visual (65%) + param (35%) predictions ─────────────────
    # Lower threshold for real images: 7-class visual classifier rarely exceeds
    # 0.5; use 0.30 so the visual signal is actually utilised.
    VISUAL_CONFIDENCE_THRESHOLD = 0.30

    if visual_pred is not None:
        visual_confidence = max(visual_proba.values()) if visual_proba else 0.0

        if visual_confidence >= VISUAL_CONFIDENCE_THRESHOLD:
            if visual_pred == param_pred:
                # Both agree — high confidence, use visual proba
                defect_type  = visual_pred
                defect_proba = visual_proba
            else:
                # Disagree — weighted fusion (65% visual, 35% param)
                all_classes = set(list(visual_proba.keys()) + list(param_proba.keys()))
                fused = {}
                for cls in all_classes:
                    vp = visual_proba.get(cls, 0.0)
                    pp = param_proba.get(cls, 0.0)
                    fused[cls] = round(0.65 * vp + 0.35 * pp, 4)
                defect_type  = max(fused, key=fused.get)
                defect_proba = fused
        else:
            # Visual not confident enough — use param-only
            defect_type  = param_pred
            defect_proba = param_proba
    else:
        # No image classifier available — fall back to param-only
        defect_type  = param_pred
        defect_proba = param_proba

    # ── Step 4d: Visual risk signal override ───────────────────────────────
    # Even when the visual label is uncertain, raw features tell us about
    # visual damage severity. Store them for the risk calculator.
    _visual_features = img_features if 'img_features' in dir() else None

    # ── Step 5: Warranty risk score ──────────────────────────────────────────
    from pipeline.models import calculate_warranty_risk
    risk_score = calculate_warranty_risk(anomaly_score, defect_type, t_feat)

    # ── Step 6: PASS / MONITOR / REJECT decision ──────────────────────────────────────
    if risk_score <= 40:
        decision = "PASS"
    elif risk_score <= 70:
        decision = "MONITOR"
    else:
        decision = "REJECT"

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "cell_id":       cell_id,
        "anomaly_score": round(anomaly_score, 4),
        "defect_type":   defect_type,
        "defect_proba":  defect_proba,
        "thermal":       t_feat,
        "risk_score":    risk_score,
        "decision":      decision,
        "latency_ms":    latency_ms,
        "image_path":    image_path,
        "audio_path":    audio_path,
        "thermal_path":  thermal_path,
        "weld_params":   {
            "voltage":    round(voltage,    2),
            "current":    round(current,    1),
            "weld_speed": round(weld_speed, 3),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch inspection
# ─────────────────────────────────────────────────────────────────────────────

def batch_inspect(cells: list, save_to_db: bool = True) -> list:
    """
    Run inspect_cell on a list of cell dicts and optionally save to DB.

    Parameters
    ----------
    cells : list of dicts, each with keys:
            cell_id, image_path, audio_path, thermal_path
            (optional: voltage, current, weld_speed, sim_defect)
    save_to_db : bool

    Returns
    -------
    list of result dicts
    """
    from pipeline.database import init_db, save_result
    if save_to_db:
        init_db()

    results = []
    for cell in cells:
        try:
            r = inspect_cell(**cell)
            if save_to_db:
                save_result(r)
            results.append(r)
            if r["decision"] == "PASS":
                status = "✓"
            elif r["decision"] == "MONITOR":
                status = "⚠"
            else:
                status = "✗"
            print(f"  {status}  {r['cell_id']:10s}  risk={r['risk_score']:3d}  {r['decision']}")
        except Exception as exc:
            print(f"  ✗  {cell.get('cell_id', '?')}  ERROR: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(ROOT, "data")

    test_cases = [
        {
            "cell_id":     "CELL-DEMO-001",
            "image_path":  os.path.join(DATA_DIR, "images",  "good_weld_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "normal_00.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "normal_0.png"),
            "sim_defect":  "good_weld",
        },
        {
            "cell_id":     "CELL-DEMO-002",
            "image_path":  os.path.join(DATA_DIR, "images",  "porosity_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "anomaly_00.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "porosity_0.png"),
            "sim_defect":  "porosity",
        },
        {
            "cell_id":     "CELL-DEMO-003",
            "image_path":  os.path.join(DATA_DIR, "images",  "burn_through_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "anomaly_03.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "cold_weld_0.png"),
            "sim_defect":  "burn_through",
        },
    ]

    print("=" * 55)
    print("  WeldSense — Pipeline Self-Test")
    print("=" * 55)
    results = batch_inspect(test_cases, save_to_db=True)
    for r in results:
        print(f"\n  {r['cell_id']}")
        print(f"    defect_type  : {r['defect_type']}")
        print(f"    anomaly_score: {r['anomaly_score']}")
        print(f"    thermal      : {r['thermal']}")
        print(f"    risk_score   : {r['risk_score']}/100")
        print(f"    decision     : {r['decision']}")
        print(f"    latency_ms   : {r['latency_ms']}")
