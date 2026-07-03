"""
models.py
=========
AI model definitions for the WeldSense inspection pipeline.

Classes / Functions
-------------------
AnomalyDetector          — IsolationForest trained on normal spectrogram images
DefectClassifier         — RandomForest trained on weld parameters CSV
calculate_warranty_risk  — Rule-based + weighted risk score 0-100
load_models()            — Load pre-trained .pkl files from models/
"""

import os
import glob
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT, "models")
AUDIO_DIR   = os.path.join(ROOT, "data", "audio")
DATA_DIR    = os.path.join(ROOT, "data")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ANOMALY DETECTOR  (acoustic modality)  — Step 8
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    One-class classifier trained on normal spectrogram images.
    Wraps sklearn IsolationForest for easier save/load.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self._fitted = False

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_spectrogram_images(folder: str, pattern: str, size: int = 64) -> np.ndarray:
        """
        Read PNG spectrogram images matching *pattern* in *folder*,
        resize to (size, size) greyscale and flatten to 1-D feature vectors.
        Returns ndarray of shape (n_samples, size*size).
        """
        from PIL import Image

        vectors = []
        paths = sorted(glob.glob(os.path.join(folder, f"*{pattern}*.png")))
        for p in paths:
            img = Image.open(p).convert("L").resize((size, size))
            vectors.append(np.array(img, dtype=np.float32).flatten() / 255.0)

        if not vectors:
            raise RuntimeError(
                f"No PNG spectrograms found matching '*{pattern}*.png' in {folder}.\n"
                "Run:  python data/generate_synthetic.py"
            )
        return np.array(vectors)

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, audio_dir: str = None, verbose: bool = True) -> dict:
        """
        Train on all normal_*.png spectrograms in audio_dir.
        Returns evaluation dict with normal/anomaly average scores.
        """
        audio_dir = audio_dir or AUDIO_DIR
        X_normal = self._load_spectrogram_images(audio_dir, "normal")

        if verbose:
            print(f"  Training AnomalyDetector on {len(X_normal)} normal samples…")

        self.model.fit(X_normal)
        self._fitted = True

        # evaluate
        scores_normal = self.model.decision_function(X_normal)
        try:
            X_anomaly = self._load_spectrogram_images(audio_dir, "anomaly")
            scores_anomaly = self.model.decision_function(X_anomaly)
            eval_info = {
                "normal_score_avg":  round(float(scores_normal.mean()),  4),
                "anomaly_score_avg": round(float(scores_anomaly.mean()), 4),
                "separation":        round(float(scores_normal.mean() - scores_anomaly.mean()), 4),
            }
        except RuntimeError:
            eval_info = {"normal_score_avg": round(float(scores_normal.mean()), 4)}

        if verbose:
            print(f"  → Eval: {eval_info}")
        return eval_info

    # ── inference ────────────────────────────────────────────────────────────

    def score(self, spectrogram: np.ndarray) -> float:
        """
        Score a single spectrogram array.
        Higher = more normal.  Negative = anomalous.

        Parameters
        ----------
        spectrogram : 2-D ndarray from sensors.audio_to_spectrogram()
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .train() or load from file.")

        flat = spectrogram.flatten()[:4096].astype(np.float32)
        # pad if needed
        if len(flat) < 4096:
            flat = np.pad(flat, (0, 4096 - len(flat)))
        flat = flat.reshape(1, -1)
        norm = flat / (np.abs(flat).max() + 1e-8)
        return float(self.model.decision_function(norm)[0])

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = None):
        import joblib
        path = path or os.path.join(MODELS_DIR, "anomaly_detector.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"  [OK] AnomalyDetector saved -> {path}")

    @classmethod
    def load(cls, path: str = None) -> "AnomalyDetector":
        import joblib
        path = path or os.path.join(MODELS_DIR, "anomaly_detector.pkl")
        detector = cls.__new__(cls)
        detector.model   = joblib.load(path)
        detector._fitted = True
        return detector


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEFECT CLASSIFIER  (weld parameter modality)  — Step 9
# ─────────────────────────────────────────────────────────────────────────────

class DefectClassifier:
    """
    Multi-class RandomForest that maps weld process parameters
    (Voltage, Current, WeldSpeed) to a defect-type label.
    """

    FEATURE_COLS = ["Voltage", "Current", "WeldSpeed"]
    TARGET_COL   = "DefectType"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=10,
        )
        self.le      = LabelEncoder()
        self._fitted = False

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, csv_path: str = None, verbose: bool = True) -> dict:
        """
        Train on welding_quality.csv.
        Returns accuracy on hold-out test set.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split

        csv_path = csv_path or os.path.join(DATA_DIR, "welding_quality.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run:  python data/generate_synthetic.py"
            )

        df = pd.read_csv(csv_path)
        X  = df[self.FEATURE_COLS].values
        y  = self.le.fit_transform(df[self.TARGET_COL].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if verbose:
            print(f"  Training DefectClassifier on {len(X_train)} samples…")
        self.clf.fit(X_train, y_train)
        self._fitted = True

        acc = self.clf.score(X_test, y_test)
        eval_info = {
            "accuracy":  round(acc, 4),
            "classes":   list(self.le.classes_),
            "n_samples": len(df),
        }
        if verbose:
            print(f"  → Accuracy: {acc:.2%}  |  Classes: {list(self.le.classes_)}")
        return eval_info

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, voltage: float, current: float, weld_speed: float) -> str:
        """
        Predict defect type from weld process parameters.
        Returns a string label, e.g. 'porosity'.
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .train() or load from file.")
        import pandas as pd
        X = np.array([[voltage, current, weld_speed]], dtype=float)
        code = self.clf.predict(X)[0]
        return str(self.le.inverse_transform([code])[0])

    def predict_proba(self, voltage: float, current: float, weld_speed: float) -> dict:
        """Return probability dict over all classes."""
        if not self._fitted:
            raise RuntimeError("Model not trained.")
        X = np.array([[voltage, current, weld_speed]], dtype=float)
        probs = self.clf.predict_proba(X)[0]
        return {cls: round(float(p), 4) for cls, p in zip(self.le.classes_, probs)}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, clf_path: str = None, enc_path: str = None):
        import joblib
        clf_path = clf_path or os.path.join(MODELS_DIR, "defect_classifier.pkl")
        enc_path = enc_path or os.path.join(MODELS_DIR, "label_encoder.pkl")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.clf, clf_path)
        joblib.dump(self.le,  enc_path)
        print(f"  [OK] DefectClassifier saved -> {clf_path}")
        print(f"  [OK] LabelEncoder saved    -> {enc_path}")

    @classmethod
    def load(cls, clf_path: str = None, enc_path: str = None) -> "DefectClassifier":
        import joblib
        clf_path = clf_path or os.path.join(MODELS_DIR, "defect_classifier.pkl")
        enc_path = enc_path or os.path.join(MODELS_DIR, "label_encoder.pkl")
        obj = cls.__new__(cls)
        obj.clf     = joblib.load(clf_path)
        obj.le      = joblib.load(enc_path)
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE DEFECT CLASSIFIER  (visual modality)  — Layer 1
# ─────────────────────────────────────────────────────────────────────────────

class ImageDefectClassifier:
    """
    Multi-class RandomForest classifier that maps visual features extracted
    from the weld photograph to a defect-type label.

    Training labels are inferred from image filename prefixes:
        spatter_00.jpg    → "spatter"
        burn_through_01.jpg → "burn_through"
        good_weld_00.jpg  → "good_weld"

    This means replacing real images in data/images/ (keeping the same
    filename prefix) automatically updates the training data.
    """

    IMAGE_CLASSES = [
        "good_weld", "burn_through", "contamination",
        "lack_of_fusion", "spatter", "porosity", "cold_weld"
    ]

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=12,
            min_samples_leaf=2,
        )
        self.le      = LabelEncoder()
        self._fitted = False

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, images_dir: str = None, verbose: bool = True) -> dict:
        """
        Train on all labelled images in images_dir.
        Labels come from the filename prefix (e.g. 'spatter' from 'spatter_00.jpg').

        Returns accuracy on hold-out test set.
        """
        import glob as _glob
        from sklearn.model_selection import train_test_split
        from pipeline.sensors import extract_image_features

        images_dir = images_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "images"
        )

        X, y_labels = [], []
        for cls in self.IMAGE_CLASSES:
            pattern = os.path.join(images_dir, f"{cls}_*.jpg")
            paths   = sorted(_glob.glob(pattern))
            if not paths:
                if verbose:
                    print(f"  [!] No images found for class '{cls}' — skipping")
                continue
            for p in paths:
                try:
                    feat = extract_image_features(p)
                    X.append(feat)
                    y_labels.append(cls)
                except Exception as e:
                    if verbose:
                        print(f"  [!] Skipping {p}: {e}")

        if len(X) < 4:
            raise RuntimeError(
                f"Not enough labelled images found in {images_dir}.\n"
                "Need at least 4 images with prefix-based names like 'spatter_00.jpg'."
            )

        X = np.array(X, dtype=np.float32)
        y = self.le.fit_transform(y_labels)

        if verbose:
            from collections import Counter
            counts = Counter(y_labels)
            print(f"  Training ImageDefectClassifier on {len(X)} images:")
            for cls, cnt in sorted(counts.items()):
                print(f"    {cls}: {cnt} images")

        # For small datasets use stratified k-fold CV for evaluation
        from collections import Counter as _Counter
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        counts = _Counter(y_labels)
        min_class_count = min(counts.values())

        # Train on all data
        self.clf.fit(X, y)
        self._fitted = True

        # Evaluate with cross-validation (handles small datasets better)
        n_splits = min(5, min_class_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf, X, y, cv=cv)
            acc = float(cv_scores.mean())
        else:
            acc = float(self.clf.score(X, y))  # fallback: training accuracy

        eval_info = {
            "accuracy":    round(acc, 4),
            "classes":     [str(c) for c in self.le.classes_],
            "n_images":    len(X),
        }

        if verbose:
            print(f"  → CV Accuracy: {acc:.2%}  |  Classes: {[str(c) for c in self.le.classes_]}")

        return eval_info

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, image_features: np.ndarray) -> str:
        """
        Predict defect type from a 20-d feature vector.
        Returns a string label e.g. 'spatter'.

        Parameters
        ----------
        image_features : np.ndarray  shape (20,)  from sensors.extract_image_features()
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call .train() or load from file.")
        feat = image_features.reshape(1, -1)
        code = self.clf.predict(feat)[0]
        return str(self.le.inverse_transform([code])[0])

    def predict_proba(self, image_features: np.ndarray) -> dict:
        """Return probability dict {class: probability} over all trained classes."""
        if not self._fitted:
            raise RuntimeError("Model not trained.")
        feat  = image_features.reshape(1, -1)
        probs = self.clf.predict_proba(feat)[0]
        return {str(cls): round(float(p), 4) for cls, p in zip(self.le.classes_, probs)}

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, clf_path: str = None, enc_path: str = None):
        import joblib
        clf_path = clf_path or os.path.join(MODELS_DIR, "image_classifier.pkl")
        enc_path = enc_path or os.path.join(MODELS_DIR, "image_label_encoder.pkl")
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.clf, clf_path)
        joblib.dump(self.le,  enc_path)
        print(f"  [OK] ImageDefectClassifier saved -> {clf_path}")
        print(f"  [OK] Image LabelEncoder saved    -> {enc_path}")

    @classmethod
    def load(cls, clf_path: str = None, enc_path: str = None) -> "ImageDefectClassifier":
        import joblib
        clf_path = clf_path or os.path.join(MODELS_DIR, "image_classifier.pkl")
        enc_path = enc_path or os.path.join(MODELS_DIR, "image_label_encoder.pkl")
        if not os.path.exists(clf_path):
            raise FileNotFoundError(
                f"Image classifier not found: {clf_path}\n"
                "Run: python pipeline/train.py"
            )
        obj = cls.__new__(cls)
        obj.clf     = joblib.load(clf_path)
        obj.le      = joblib.load(enc_path)
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WARRANTY RISK SCORE  — Step 10
# ─────────────────────────────────────────────────────────────────────────────

# Severity multiplier for each defect type (0 = safe, 1 = certain failure)
DEFECT_RISK_WEIGHTS = {
    "good_weld":      0.00,
    "normal":         0.00,
    "spatter":        0.30,
    "contamination":  0.45,
    "lack_of_fusion": 0.60,
    "misalignment":   0.60,
    "burn_through":   0.70,
    "porosity":       0.80,   # internal void → structural weakness
    "cold_weld":      0.90,   # weak mechanical bond → likely field failure
}


def calculate_warranty_risk(
    anomaly_score: float,
    defect_type: str,
    thermal_features: dict,
) -> int:
    """
    Combine multimodal signals into a single warranty risk score 0-100.

    Parameters
    ----------
    anomaly_score    : float  output of IsolationForest.decision_function()
                              negative = anomalous, positive = normal
    defect_type      : str    label from DefectClassifier
    thermal_features : dict   {"peak": float, "mean": float, "std": float}
                              from sensors.read_thermal()

    Returns
    -------
    int  0-100  (100 = near-certain warranty failure)
    """
    # ── acoustic component (base risk) ─────────────────────────────────
    # IsolationForest scores typically in [-0.5, +0.5]
    # Map: score=+0.3 → risk=0,  score=-0.3 → risk~90
    base_risk = max(0.0, min(100.0, (-anomaly_score + 0.3) * 150.0))

    # ── defect type multiplier ──────────────────────────────────────────
    multiplier = DEFECT_RISK_WEIGHTS.get(defect_type.lower(), 0.5)

    # ── thermal penalty ─────────────────────────────────────────────────
    # std > 0.20 indicates significant heating unevenness
    thermal_penalty = min(20, int(thermal_features.get("std", 0) * 30))

    # ── peak temperature bonus ──────────────────────────────────────────
    peak = thermal_features.get("peak", 0)
    peak_penalty = min(10, int(max(0, peak - 0.75) * 40))

    # ── weighted combination ────────────────────────────────────────────
    risk = (
        base_risk   * (0.50 + multiplier * 0.50)   # acoustic × defect weight
        + thermal_penalty                            # thermal unevenness
        + peak_penalty                               # extreme peak temperature
    )

    return max(0, min(100, int(round(risk))))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CONVENIENCE: load_models()
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    """
    Load both pre-trained models from the models/ directory.

    Returns
    -------
    (AnomalyDetector, DefectClassifier)

    Raises
    ------
    FileNotFoundError if pkl files are missing — call train.py first.
    """
    detector   = AnomalyDetector.load()
    classifier = DefectClassifier.load()
    return detector, classifier


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing calculate_warranty_risk…")
    # Normal weld — should be LOW risk
    r1 = calculate_warranty_risk(
        anomaly_score=0.25,
        defect_type="good_weld",
        thermal_features={"peak": 0.55, "mean": 0.38, "std": 0.05},
    )
    print(f"  Normal  → risk={r1}/100  (expect <20)")

    # Porosity weld — should be HIGH risk
    r2 = calculate_warranty_risk(
        anomaly_score=-0.15,
        defect_type="porosity",
        thermal_features={"peak": 0.85, "mean": 0.48, "std": 0.22},
    )
    print(f"  Porosity → risk={r2}/100  (expect >60)")

    # Cold weld — should be VERY HIGH risk
    r3 = calculate_warranty_risk(
        anomaly_score=-0.30,
        defect_type="cold_weld",
        thermal_features={"peak": 0.60, "mean": 0.30, "std": 0.18},
    )
    print(f"  Cold weld → risk={r3}/100  (expect >70)")
