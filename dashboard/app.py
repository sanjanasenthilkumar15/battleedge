"""
app.py — WeldSense Live Inspection Dashboard
=============================================
Run from the project root:
    streamlit run dashboard/app.py

Implements Steps 13-16 of the implementation plan:
  Step 13 — Streamlit inspection UI + PASS/REJECT result
  Step 14 — 3-column sensor output (image, thermal, spectrogram + audio)
  Step 15 — Plotly risk gauge
  Step 16 — Inspection history table with summary metrics
"""

import os
import sys
import glob
import random
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── path setup so we can import pipeline from the project root ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.database    import init_db, save_result, load_history, get_summary
from pipeline.pipeline_core import inspect_cell, _load_models

def apply_theme(theme):
    if theme == "Dark":
        bg = "#0d1117"
        sidebar_bg = "#161b22"
        text = "#e6edf3"
        subtext = "#c9d1d9"
        card_bg = "#1c2128"
        border = "#30363d"
    else:
        bg = "#F8FAFC"
        sidebar_bg = "#EEF2FF"
        text = "#000000"
        subtext = "#000000"
        card_bg = "#FFFFFF"
        border = "#E2E8F0"

    st.markdown(f"""
    <style>
    /* Page background */
    .stApp, .stApp > div {{
        background-color: {bg} !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        width: 220px !important;
        min-width: 220px !important;
    }}

    /* Force sidebar to NEVER collapse */
    [data-testid="stSidebar"][aria-expanded="false"] {{
        width: 220px !important;
        min-width: 220px !important;
        display: block !important;
    }}
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}

    /* ALL text everywhere — no exceptions */
    html, body, p, span, div, label, h1, h2, h3, h4, h5, h6,
    li, a, td, th, caption, small, strong, em,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stTextInput label, .stSelectbox label,
    .stRadio label, .stCheckbox label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    [data-testid="stCaptionContainer"],
    [data-testid="stText"],
    .stCaption, .stCaption p,
    [class*="css"] p, [class*="css"] span,
    [class*="css"] label, [class*="css"] div {{
        color: {text} !important;
    }}

    /* Tabs — show text and icons */
    [data-testid="stTabs"] button {{
        color: {text} !important;
        background: transparent !important;
        opacity: 1 !important;
    }}
    [data-testid="stTabs"] button p,
    [data-testid="stTabs"] button span {{
        color: {text} !important;
        opacity: 1 !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }}

    /* Subtitle line under header */
    .stMarkdown p:first-child {{
        color: {subtext} !important;
        font-weight: 500 !important;
    }}

    /* Dataframe / table */
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    .dataframe td, .dataframe th {{
        color: {text} !important;
        background-color: {card_bg} !important;
    }}

    /* Expander */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {{
        color: {text} !important;
    }}
    [data-testid="stExpander"] > div {{
        background-color: {card_bg} !important;
    }}

    /* Input boxes */
    .stTextInput input, .stSelectbox select,
    [data-testid="stTextInput"] input {{
        color: {text} !important;
        background-color: {card_bg} !important;
        border-color: {border} !important;
    }}

    /* Section header caps */
    .stMarkdown p[style*="uppercase"],
    small {{ color: {subtext} !important; }}

    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data paths
# ─────────────────────────────────────────────────────────────────────────────
IMG_DIR     = os.path.join(ROOT, "data", "images")
THERMAL_DIR = os.path.join(ROOT, "data", "thermal")
AUDIO_DIR   = os.path.join(ROOT, "data", "audio")

# Defect → (image_prefix, thermal_prefix, audio_type)
DEFECT_MAP = {
    "good_weld":      ("good_weld",    "normal",      "normal"),
    "porosity":       ("porosity",     "porosity",    "anomaly"),
    "burn_through":   ("burn_through", "misalignment","anomaly"),
    "contamination":  ("contamination","normal",      "anomaly"),
    "lack_of_fusion": ("lack_of_fusion","cold_weld",  "anomaly"),
    "spatter":        ("spatter",      "normal",      "anomaly"),
    "cold_weld":      ("good_weld",   "cold_weld",    "anomaly"),
    "misalignment":   ("good_weld",   "misalignment", "anomaly"),
}

DEFECT_LABELS = list(DEFECT_MAP.keys())


def _pick_file(directory: str, prefix: str, ext: str) -> str | None:
    """Return a random file matching directory/prefix_*.ext, or None."""
    pattern = os.path.join(directory, f"{prefix}_*.{ext}")
    matches = glob.glob(pattern)
    return random.choice(matches) if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeldSense — EV Weld Inspection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Init DB & Warmup Models
# ─────────────────────────────────────────────────────────────────────────────
init_db()
# Warm up models on app start to avoid latency spike on first run
_load_models()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    theme = st.radio("🎨 Theme", ["Light", "Dark"], horizontal=True)
    apply_theme(theme)

    st.markdown("## ⚡ WeldSense")
    st.markdown("*EV Battery Cell Weld Inspector*")
    st.divider()

    st.markdown("### 🔧 Inspection Setup")
    cell_id = st.text_input("Cell ID", value="CELL-001", key="cell_id_input")

    sim_defect = st.selectbox(
        "Simulate defect type",
        DEFECT_LABELS,
        format_func=lambda x: x.replace("_", " ").title(),
        key="defect_selector",
    )

    st.markdown("#### Process Parameters *(optional)*")
    with st.expander("Override weld parameters"):
        voltage    = st.slider("Voltage (V)",   15.0, 35.0, 22.0, 0.5)
        current    = st.slider("Current (A)",   100.0, 300.0, 180.0, 5.0)
        weld_speed = st.slider("Weld Speed (m/s)", 0.2, 1.0, 0.5, 0.05)
        use_custom = st.checkbox("Use these values", value=False)

    run_btn = st.button("▶  Run Inspection", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 🗄 Database")
    col_db1, col_db2 = st.columns(2)
    summary = get_summary()
    col_db1.metric("Total", summary["total"])
    col_db2.metric("Rejected", summary["n_reject"])

    if st.button("🗑 Reset Database", use_container_width=True):
        from pipeline.database import clear_db
        clear_db()
        st.success("Database cleared.")
        st.rerun()

    st.divider()
    st.caption("Built for Tata Technologies InnoVent 2026–27")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='
    color: #1A56DB;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
'>⚡ WeldSense — AI Weld Inspection</h1>
<p style='color:#000000; font-size:0.9rem; margin-top:0;'>
  Edge-AI warranty risk scoring · Tata Motors Nexon EV production line
</p>
""", unsafe_allow_html=True)

tab_inspect, tab_history, tab_analytics = st.tabs(
    ["🔬 Inspect Cell", "📋 Inspection History", "📊 Analytics"]
)


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 1: INSPECT CELL ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_inspect:

    # ── Run inspection ────────────────────────────────────────────────────────
    if run_btn:
        img_prefix, th_prefix, aud_type = DEFECT_MAP[sim_defect]

        def_type = sim_defect.replace('_', ' ').title()
        weld_images = {
            "Good Weld": os.path.join(IMG_DIR, "good_weld_00.jpg"),
            "Burn Through": os.path.join(IMG_DIR, "burn_through_00.jpg"),
            "Lack Of Fusion": os.path.join(IMG_DIR, "lack_of_fusion_00.jpg"),
            "Porosity": os.path.join(IMG_DIR, "porosity_00.jpg"),
            "Spatter": os.path.join(IMG_DIR, "spatter_00.jpg"),
            "Contamination": os.path.join(IMG_DIR, "contamination_00.jpg"),
            "Cold Weld": os.path.join(IMG_DIR, "good_weld_00.jpg"),
            "Misalignment": os.path.join(IMG_DIR, "good_weld_00.jpg")
        }
        image_file   = weld_images.get(def_type, weld_images["Good Weld"])
        
        thermal_file = _pick_file(THERMAL_DIR, th_prefix,  "png")
        audio_file   = _pick_file(AUDIO_DIR,   aud_type,   "wav")

        missing = [n for n, f in [("image", image_file),
                                   ("thermal", thermal_file),
                                   ("audio", audio_file)] if f is None]
        if missing:
            st.error(
                f"Missing data files for: {', '.join(missing)}.\n\n"
                "Please run:  `python data/generate_synthetic.py`"
            )
            st.stop()

        kwargs = dict(
            cell_id      = cell_id,
            image_path   = image_file,
            audio_path   = audio_file,
            thermal_path = thermal_file,
            sim_defect   = sim_defect,
        )
        if use_custom:
            kwargs.update(voltage=voltage, current=current, weld_speed=weld_speed)

        with st.spinner("Running WeldSense multi-sensor pipeline…"):
            try:
                result = inspect_cell(**kwargs)
            except FileNotFoundError as e:
                st.error(str(e))
                st.info("Run `python pipeline/train.py` to train the models first.")
                st.stop()

        save_result(result)

        # ── Row 1: Decision banner ────────────────────────────────────────────
        decision = result["decision"]
        if decision == "PASS":
            st.success(f"✅ PASS — Cell {result['cell_id']} — Low Risk")
        elif decision == "MONITOR":
            st.warning(f"⚠️ MONITOR — Cell {result['cell_id']} — Moderate Risk")
        else:
            st.error(f"❌ REJECT — Cell {result['cell_id']} — High Warranty Risk")

        # ── Row 2: Risk Gauge ─────────────────────────────────────────────────
        score = result["risk_score"]

        def risk_gauge(score: int):
            color = "#16A34A" if score <= 40 else ("#D97706" if score <= 70 else "#DC2626")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "Warranty Risk Score", "font": {"color": "#000000", "size": 14}},
                number={"font": {"color": color, "size": 48}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#E2E8F0",
                        "tickfont": {"color": "#000000", "size": 11},
                    },
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "rgba(22,163,74,0.10)"},
                        {"range": [40, 70], "color": "rgba(217,119,6,0.10)"},
                        {"range": [70,100], "color": "rgba(220,38,38,0.12)"},
                    ],
                    "threshold": {
                        "line": {"color": "#DC2626", "width": 3},
                        "thickness": 0.85,
                        "value": 70,
                    },
                },
            ))
            fig.update_layout(
                height=380,
                margin=dict(t=40, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            return fig

        st.plotly_chart(risk_gauge(score), use_container_width=True)

        # ── Row 3: 4-Metric Grid ──────────────────────────────────────────────
        st.divider()
        st.caption("INSPECTION RESULTS")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score", f"{result['risk_score']} / 100")
        c2.metric("Defect Type", result["defect_type"].replace("_", " ").title())
        
        with c3:
            st.metric("Latency", f"{result['latency_ms']} ms")
            lat = result['latency_ms']
            if lat < 200:
                st.success(f"⚡ {lat} ms")
            elif lat <= 400:
                st.warning(f"⚡ {lat} ms")
            else:
                st.error(f"⚡ {lat} ms")
                
        c4.metric("Anomaly Score", f"{result['anomaly_score']:.3f}")

        # ── Row 4: Thermal Grid ───────────────────────────────────────────────
        st.divider()
        st.caption("THERMAL FEATURES")
        t1, t2, t3 = st.columns(3)
        t1.metric("Peak",  f"{result['thermal']['peak']:.3f}")
        t2.metric("Mean",  f"{result['thermal']['mean']:.3f}")
        t3.metric("Std",   f"{result['thermal']['std']:.3f}",
                   delta="⚠ uneven" if result["thermal"]["std"] > 0.20 else "✓ even",
                   delta_color="inverse")

        # ── Row 5: Defect Probabilities ───────────────────────────────────────
        st.divider()
        st.caption("DEFECT PROBABILITIES")
        proba_df = (
            pd.Series(result["defect_proba"])
            .sort_values(ascending=False)
            .head(4)
            .reset_index()
        )
        proba_df.columns = ["Defect", "Probability"]
        proba_df["Defect"] = proba_df["Defect"].str.replace("_", " ").str.title()
        fig_proba = px.bar(
            proba_df, x="Probability", y="Defect", orientation="h",
            color="Probability",
            color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
            range_x=[0, 1],
        )
        fig_proba.update_layout(
            height=180,
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(color="#000000", size=11)),
            xaxis=dict(tickfont=dict(color="#000000", size=10),
                       gridcolor="#E2E8F0"),
            font={"family": "Inter", "color": "#000000"},
        )
        st.plotly_chart(fig_proba, use_container_width=True)

        # ── Step 14: Sensor Outputs ───────────────────────────────────────────
        st.divider()
        st.caption("SENSOR OUTPUTS")

        if decision == "PASS":
            st.success(f"✅ PASS — Cell {result['cell_id']} — Low Risk")
        elif decision == "MONITOR":
            st.warning(f"⚠️ MONITOR — Cell {result['cell_id']} — Moderate Risk")
        else:
            st.error(f"❌ REJECT — Cell {result['cell_id']} — High Warranty Risk")
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📷 Camera (Visual)**")
            st.image(image_file, use_container_width=True)
            
            def_type = sim_defect.replace('_',' ').title()
            if sim_defect == "good_weld":
                def_color = "#16A34A"
            elif sim_defect in ["spatter", "contamination"]:
                def_color = "#D97706"
            else:
                def_color = "#DC2626"
            
            st.markdown(f"<div style='font-size:0.85rem; color:#000000; margin-top:0.5rem;'>Weld image — <span style='color:{def_color}'><b>{def_type}</b></span></div>", unsafe_allow_html=True)

        with c2:
            st.markdown("**🌡 IR Sensor (Thermal)**")
            st.image(thermal_file,
                     caption=f"Heat map  std={result['thermal']['std']:.3f}",
                     use_container_width=True)
            st.caption("IR thermal map — heat distribution used by root cause classifier (Layer 3)")

        with c3:
            st.markdown("**🎤 Microphone (Acoustic)**")
            spec_file = audio_file.replace(".wav", ".png")
            if os.path.exists(spec_file):
                st.image(spec_file, use_container_width=True)
                st.caption("Mel spectrogram — frequency pattern used by anomaly detection model (Layer 2)")
            else:
                st.warning("Spectrogram PNG not found.")
            st.audio(audio_file)

        # ── Weld parameters used ──────────────────────────────────────────────
        with st.expander("🔩 Weld Parameters Used"):
            wp = result["weld_params"]
            p1, p2, p3 = st.columns(3)
            p1.metric("Voltage",    f"{wp['voltage']} V")
            p2.metric("Current",    f"{wp['current']} A")
            p3.metric("Weld Speed", f"{wp['weld_speed']} m/s")

    else:
        # ── Placeholder when no inspection has been run ───────────────────────
        st.markdown("""
<div style='
    text-align:center;
    padding: 4rem 2rem;
    color: #4a5568;
    border: 2px dashed #2a3050;
    border-radius: 16px;
    margin-top: 2rem;
'>
    <div style='font-size:3rem;margin-bottom:1rem;'>⚡</div>
    <div style='font-size:1.1rem;font-weight:600;color:#6070a0;'>
        Select a defect type and press ▶ Run Inspection
    </div>
    <div style='font-size:0.85rem;margin-top:0.5rem;'>
        The pipeline will read all 3 sensor modalities and compute a warranty risk score.
    </div>
</div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 2: INSPECTION HISTORY  — Step 16 ─────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    df = load_history(n=100)

    if df.empty:
        st.info("No inspections yet — run your first inspection in the Inspect Cell tab.")
    else:
        # ── Summary metrics ───────────────────────────────────────────────────
        st.divider()
        st.caption("SUMMARY")
        s = get_summary()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Inspected",  s["total"])
        c2.metric("Rejected",         s["n_reject"],
                  delta=f"{s['rejection_rate_pct']:.0f}% rejection rate",
                  delta_color="inverse")
        c3.metric("Avg Risk Score",   f"{s['avg_risk']:.1f}")
        c4.metric("Avg Latency",      f"{s['avg_latency_ms']:.0f} ms")

        # ── Risk trend sparkline ──────────────────────────────────────────────
        st.divider()
        st.caption("RISK SCORE OVER TIME")
        df_chart = df.sort_values("id")[["id", "risk_score", "decision"]].copy()
        df_chart["colour"] = df_chart["decision"].map(
            {"PASS": "#16A34A", "MONITOR": "#D97706", "REJECT": "#DC2626"}
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_chart["id"],
            y=df_chart["risk_score"],
            mode="lines+markers",
            line=dict(color="#1A56DB", width=2),
            marker=dict(
                color=df_chart["colour"],
                size=8,
                line=dict(color="#FFFFFF", width=1),
            ),
            name="Risk Score",
        ))
        fig_trend.add_hline(y=70, line_dash="dash", line_color="#DC2626",
                            annotation_text="Reject threshold (70)",
                            annotation_font_color="#DC2626")
        fig_trend.update_layout(
            height=220,
            margin=dict(t=10, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Inspection #", gridcolor="#E2E8F0",
                       tickfont=dict(color="#000000")),
            yaxis=dict(title="Risk Score", range=[0, 105],
                       gridcolor="#E2E8F0",
                       tickfont=dict(color="#000000")),
            font={"family": "Inter", "color": "#000000"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Styled table ──────────────────────────────────────────────────────
        st.divider()
        st.caption("ALL RECORDS")

        display_cols = ["cell_id", "defect_type", "risk_score", "decision", "latency_ms", "timestamp"]
        df_display = df[display_cols].copy()
        df_display["defect_type"] = (
            df_display["defect_type"].str.replace("_", " ").str.title()
        )
        df_display.columns = ["Cell ID", "Defect Type", "Risk Score", "Decision", "Latency (ms)", "Timestamp"]

        def colour_decision(val):
            color = "#16A34A" if val == "PASS" else ("#D97706" if val == "MONITOR" else "#DC2626")
            return f"color: {color}; font-weight: bold;"

        try:
            styled = df_display.style.map(colour_decision, subset=["Decision"])
        except AttributeError:
            styled = df_display.style.applymap(colour_decision, subset=["Decision"])
            
        styled = styled.format({"Risk Score": "{:.0f}"})
        st.dataframe(styled, use_container_width=True, height=400)

        # ── Download button ───────────────────────────────────────────────────
        csv_bytes = df_display.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download History CSV",
            data=csv_bytes,
            file_name=f"battleedge_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ── TAB 3: ANALYTICS ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with tab_analytics:
    df = load_history(n=999_999)

    if df.empty or len(df) < 2:
        st.info("Run at least 2 inspections to see analytics.")
    else:
        s = get_summary()

        col_l, col_r = st.columns(2)

        # ── Defect distribution donut ─────────────────────────────────────────
        with col_l:
            st.divider()
            st.caption("DEFECT TYPE DISTRIBUTION")
            defect_counts = df["defect_type"].value_counts()
            color_map = {
                "Good Weld": "#16A34A",
                "Burn Through": "#DC2626",
                "Lack Of Fusion": "#D97706",
                "Contamination": "#7C3AED",
                "Porosity": "#0284C7",
                "Spatter": "#EA580C",
                "Cold Weld": "#DB2777"
            }
            donut_labels = [x.replace("_", " ").title() for x in defect_counts.index]
            fig_donut = go.Figure(go.Pie(
                labels=donut_labels,
                values=defect_counts.values,
                hole=0.55,
                marker_colors=[color_map.get(lbl, "#94A3B8") for lbl in donut_labels],
                textfont=dict(color="#000000"),
            ))
            fig_donut.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(color="#000000", size=11)),
                font={"family": "Inter"},
                annotations=[dict(
                    text=f"<b>{len(df)}</b><br>cells",
                    font=dict(size=18, color="#000000", family="Inter"),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Risk score distribution histogram ─────────────────────────────────
        with col_r:
            st.divider()
            st.caption("RISK SCORE DISTRIBUTION")
            fig_hist = go.Figure(go.Histogram(
                x=df["risk_score"],
                nbinsx=20,
                marker_color="#1A56DB",
                opacity=0.8,
            ))
            fig_hist.add_vline(x=70, line_dash="dash", line_color="#DC2626",
                               annotation_text="Threshold",
                               annotation_font_color="#DC2626")
            fig_hist.update_layout(
                height=280,
                margin=dict(t=10, b=30, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Risk Score", gridcolor="#E2E8F0",
                           tickfont=dict(color="#000000")),
                yaxis=dict(title="Count",     gridcolor="#E2E8F0",
                           tickfont=dict(color="#000000")),
                font={"family": "Inter", "color": "#000000"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(f"Based on n={len(df)} simulated inspection runs · threshold line at score 70")

        # ── Risk by defect type (box plot) ────────────────────────────────────
        st.divider()
        st.caption("RISK SCORE BY DEFECT TYPE")
        df_plot = df.copy()
        df_plot["defect_label"] = (
            df_plot["defect_type"].str.replace("_", " ").str.title()
        )
        color_map = {
            "Good Weld": "#16A34A",
            "Burn Through": "#DC2626",
            "Lack Of Fusion": "#D97706",
            "Contamination": "#7C3AED",
            "Porosity": "#0284C7",
            "Spatter": "#EA580C",
            "Cold Weld": "#DB2777"
        }
        fig_box = px.box(
            df_plot, x="defect_label", y="risk_score",
            color="defect_label",
            color_discrete_map=color_map,
            labels={"defect_label": "Defect Type", "risk_score": "Risk Score"},
        )
        fig_box.add_hline(y=70, line_dash="dash", line_color="#DC2626")
        fig_box.update_layout(
            height=320,
            margin=dict(t=10, b=40, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#000000"), gridcolor="#E2E8F0"),
            yaxis=dict(tickfont=dict(color="#000000"), gridcolor="#E2E8F0"),
            legend=dict(font=dict(color="#000000")),
            font={"family": "Inter", "color": "#000000"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Thermal std vs risk scatter ───────────────────────────────────────
        st.divider()
        st.caption("THERMAL UNIFORMITY VS RISK SCORE")
        df_th = df.dropna(subset=["thermal_std", "risk_score"])
        if not df_th.empty:
            fig_scatter = px.scatter(
                df_th,
                x="thermal_std", y="risk_score",
                color="decision",
                color_discrete_map={"PASS": "#16A34A", "MONITOR": "#D97706", "REJECT": "#DC2626"},
                hover_data=["cell_id", "defect_type"],
                labels={"thermal_std": "Thermal Std Dev (unevenness)",
                        "risk_score":  "Risk Score"},
                trendline="ols",
            )
            fig_scatter.add_hline(y=70, line_dash="dash", line_color="#DC2626")
            fig_scatter.update_layout(
                height=300,
                margin=dict(t=10, b=40, l=40, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#000000"),
                           gridcolor="#E2E8F0"),
                yaxis=dict(tickfont=dict(color="#000000"),
                           gridcolor="#E2E8F0"),
                legend=dict(font=dict(color="#000000")),
                font={"family": "Inter", "color": "#000000"},
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Key insights table ────────────────────────────────────────────────
        st.divider()
        st.caption("KEY INSIGHTS")
        insights = []
        for defect, cnt in s["defect_counts"].items():
            sub = df[df["defect_type"] == defect]
            avg_risk = sub["risk_score"].mean()
            reject_r = (sub["decision"] == "REJECT").mean() * 100
            insights.append({
                "Defect Type":    defect.replace("_", " ").title(),
                "Count":          int(cnt),
                "Avg Risk":       f"{avg_risk:.1f}",
                "Rejection Rate": f"{reject_r:.0f}%",
            })
        if insights:
            ins_df = pd.DataFrame(insights).sort_values("Count", ascending=False)
            st.dataframe(ins_df, use_container_width=True, hide_index=True)
