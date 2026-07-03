"""
database.py
===========
SQLite persistence layer for WeldSense inspection results.

Functions
---------
init_db()          — Create the inspections table (idempotent)
save_result(dict)  — Insert one inspection record
load_history(n)    — Fetch last n records as a pandas DataFrame
get_summary()      — Aggregate statistics dict
clear_db()         — Delete all records (for testing)
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "inspections.db")


# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS inspections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_id       TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    anomaly_score REAL,
    defect_type   TEXT,
    thermal_peak  REAL,
    thermal_mean  REAL,
    thermal_std   REAL,
    risk_score    INTEGER,
    decision      TEXT,
    latency_ms    INTEGER,
    image_path    TEXT,
    audio_path    TEXT,
    thermal_path  TEXT
)
"""


def init_db(db_path: str = DB_PATH) -> None:
    """Create the inspections table if it does not already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE)
        conn.commit()
    print(f"[DB] Initialised -> {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def save_result(result: dict, db_path: str = DB_PATH) -> int:
    """
    Persist a single inspection result to the database.

    Parameters
    ----------
    result  : dict  output of pipeline.inspect.inspect_cell()
    db_path : str   path to the SQLite file

    Returns
    -------
    int  newly inserted row id
    """
    thermal = result.get("thermal", {})
    row = (
        result.get("cell_id",       "UNKNOWN"),
        datetime.now().isoformat(timespec="seconds"),
        result.get("anomaly_score", None),
        result.get("defect_type",   None),
        thermal.get("peak",         None),
        thermal.get("mean",         None),
        thermal.get("std",          None),
        result.get("risk_score",    None),
        result.get("decision",      None),
        result.get("latency_ms",    None),
        result.get("image_path",    None),
        result.get("audio_path",    None),
        result.get("thermal_path",  None),
    )

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("""
            INSERT INTO inspections
                (cell_id, timestamp, anomaly_score, defect_type,
                 thermal_peak, thermal_mean, thermal_std,
                 risk_score, decision, latency_ms,
                 image_path, audio_path, thermal_path)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)
        conn.commit()
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def load_history(n: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Return the last *n* inspections as a DataFrame.
    Returns an empty DataFrame if the table is empty or DB doesn't exist.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            f"SELECT * FROM inspections ORDER BY id DESC LIMIT {int(n)}",
            conn,
        )
    return df


def get_summary(db_path: str = DB_PATH) -> dict:
    """
    Compute aggregate statistics over all inspections.

    Returns
    -------
    dict with keys:
        total, n_pass, n_reject, rejection_rate_pct,
        avg_risk, avg_latency_ms,
        defect_counts (dict)
    """
    df = load_history(n=999_999, db_path=db_path)
    if df.empty:
        return {
            "total": 0,
            "n_pass": 0,
            "n_reject": 0,
            "rejection_rate_pct": 0.0,
            "avg_risk": 0.0,
            "avg_latency_ms": 0.0,
            "defect_counts": {},
        }

    n_reject = int((df["decision"] == "REJECT").sum())
    return {
        "total":               len(df),
        "n_pass":              int((df["decision"] == "PASS").sum()),
        "n_reject":            n_reject,
        "rejection_rate_pct":  round(n_reject / len(df) * 100, 1),
        "avg_risk":            round(float(df["risk_score"].mean()), 1),
        "avg_latency_ms":      round(float(df["latency_ms"].mean()), 1),
        "defect_counts":       df["defect_type"].value_counts().to_dict(),
    }


def clear_db(db_path: str = DB_PATH) -> None:
    """Delete ALL inspection records. Use for testing/demo resets."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM inspections")
        conn.commit()
    print(f"[DB] Cleared all records from {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()

    dummy = {
        "cell_id":       "TEST-001",
        "anomaly_score": -0.12,
        "defect_type":   "porosity",
        "thermal":       {"peak": 0.82, "mean": 0.45, "std": 0.20},
        "risk_score":    74,
        "decision":      "REJECT",
        "latency_ms":    312,
        "image_path":    "data/images/porosity_00.jpg",
        "audio_path":    "data/audio/anomaly_00.wav",
        "thermal_path":  "data/thermal/porosity_0.png",
    }

    row_id = save_result(dummy)
    print(f"[DB] Inserted row id={row_id}")

    df = load_history(n=10)
    print(df.to_string(index=False))

    summary = get_summary()
    print(f"\n[DB] Summary: {summary}")
