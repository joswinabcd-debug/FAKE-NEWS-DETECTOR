"""
TRUTHSCAN AI - Web Application
Deep Learning-Based Fake News Detection Using CNN and LSTM
Premium Dark AI Theme with Metallic Accents and Emerald Styling.
"""

import os
import json
import time
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import config
from src.dataset import get_dataset_statistics, check_dataset_exists
from src.train import train_pipeline
from src.evaluate import evaluate_models
from src.predict import NewsPredictor
from src.download_data import ensure_dataset

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TRUTHSCAN AI - Fake News Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Navigation State Safety Buffer (Phase 3 Fix)
# Prevents StreamlitWidgetAlreadyInstantiatedError
# -----------------------------------------------------------------------------
NAV_PAGES = [
    "HOME",
    "FAKE NEWS DETECTOR",
    "MODEL COMPARISON",
    "TRAINING SESSION",
    "SESSIONS"
]

# Consume any programmatic navigation target before the widget is created
if "pending_nav" in st.session_state and st.session_state.pending_nav:
    st.session_state.current_nav = st.session_state.pop("pending_nav")

if "current_nav" not in st.session_state or st.session_state.current_nav not in NAV_PAGES:
    st.session_state.current_nav = "HOME"

# Initialize persistent session history containers if not present
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_training_run" not in st.session_state:
    st.session_state.last_training_run = None

if "sessions_tab" not in st.session_state:
    st.session_state.sessions_tab = "Prediction History"


def navigate_to(page_name: str):
    """Safely queues page navigation without mutating widget-bound state after instantiation."""
    st.session_state.pending_nav = page_name
    st.rerun()


# -----------------------------------------------------------------------------
# Dark AI Dashboard Theme CSS (Phases 10-15 & 24)
# Metallic gunmetal surfaces, emerald green (#00E676) accents, controlled glow
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global reset and dark theme base */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #050807 !important;
        color: #FFFFFF !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1220px !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    /* -------------------------------------------------------------------------
       TOP NAVIGATION TILES (Metallic Dark + Emerald Glow)
       ------------------------------------------------------------------------- */
    div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"],
    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0.4rem 0 1.2rem 0 !important;
        margin-bottom: 1.4rem !important;
        border-bottom: 1px solid #14281D !important;
    }

    div[data-testid="stRadioGroup"] > div {
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
    }

    /* Navigation Tile Design */
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"],
    label[data-testid="stRadioOption"],
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: linear-gradient(180deg, #14221A 0%, #0D1712 55%, #080F0C 100%) !important;
        border: 1px solid #1B3828 !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.25rem !important;
        cursor: pointer !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        min-width: 170px !important;
        margin: 0 !important;
    }

    /* Hide native radio button graphics safely */
    label[data-testid="stRadioOption"] input {
        opacity: 0 !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }
    label[data-testid="stRadioOption"] svg,
    label[data-testid="stRadioOption"] div[class*="e1mpz0hj4"],
    label[data-testid="stRadioOption"] div[class*="e1mpz0hj5"],
    label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]),
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Text inside Tile */
    label[data-testid="stRadioOption"] > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }
    label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"],
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        text-align: center !important;
        margin: 0 !important;
    }
    label[data-testid="stRadioOption"] p,
    div[data-testid="stRadio"] label[data-baseweb="radio"] p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        color: #A7B5AD !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
        text-align: center !important;
    }

    /* Tile Hover Effect (Lift + Subtle Emerald Glow) */
    label[data-testid="stRadioOption"]:hover,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        transform: translateY(-2px) !important;
        border-color: #00E676 !important;
        background: linear-gradient(180deg, #1C3325 0%, #122218 55%, #0A140E 100%) !important;
        box-shadow: 0 4px 14px rgba(0, 230, 118, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    label[data-testid="stRadioOption"]:hover p,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover p {
        color: #FFFFFF !important;
    }

    /* Active Tile */
    label[data-testid="stRadioOption"][data-selected="true"],
    div[data-testid="stRadioGroup"] div[data-selected="true"] label[data-testid="stRadioOption"],
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(180deg, #1B3B28 0%, #10241A 55%, #0B1912 100%) !important;
        border: 1.5px solid #00E676 !important;
        box-shadow: 0 0 16px rgba(0, 230, 118, 0.22), inset 0 1px 0 rgba(0, 230, 118, 0.2) !important;
        transform: translateY(0) !important;
    }
    label[data-testid="stRadioOption"][data-selected="true"] p,
    div[data-testid="stRadioGroup"] div[data-selected="true"] label[data-testid="stRadioOption"] p,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: #00E676 !important;
        font-weight: 800 !important;
    }

    /* -------------------------------------------------------------------------
       METALLIC HERO CARD & TILES
       ------------------------------------------------------------------------- */
    .metallic-hero-card {
        background: linear-gradient(135deg, #0F1B14 0%, #0B140F 45%, #070D0A 100%);
        border: 1px solid #1B3828;
        border-radius: 12px;
        padding: 2.4rem 2.6rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
    }
    .metallic-hero-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00B85A 0%, #00E676 50%, #00B85A 100%);
        border-radius: 12px 12px 0 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0, 230, 118, 0.12);
        border: 1px solid rgba(0, 230, 118, 0.3);
        color: #00E676;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    .hero-title-main {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.15;
        color: #FFFFFF;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle-secondary {
        font-size: 1.15rem;
        font-weight: 600;
        color: #00E676;
        margin: 0 0 1rem 0;
    }
    .hero-divider {
        width: 50px;
        height: 2px;
        background: #00E676;
        margin: 1rem 0;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #A7B5AD;
        line-height: 1.6;
        max-width: 860px;
        margin: 0;
    }

    /* -------------------------------------------------------------------------
       PREMIUM DARK METALLIC CARDS
       ------------------------------------------------------------------------- */
    .premium-card {
        background: #0B120E;
        border: 1px solid #1B3828;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .card-header-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #00E676;
        margin-bottom: 0.4rem;
    }
    .card-val-display {
        font-size: 1.8rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .card-desc-text {
        font-size: 0.88rem;
        color: #A7B5AD;
        line-height: 1.55;
        margin-top: 0.4rem;
    }

    /* Flowchart Tile Nodes */
    .flow-step-card {
        background: linear-gradient(180deg, #101B15 0%, #0A120E 100%);
        border: 1px solid #1B3828;
        border-radius: 8px;
        padding: 0.85rem 0.6rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        min-height: 84px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .flow-step-card:hover {
        border-color: #00E676;
        transform: translateY(-2px);
    }
    .flow-step-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.15rem;
    }
    .flow-step-sub {
        font-size: 0.72rem;
        color: #A7B5AD;
        font-weight: 500;
    }
    .flow-arrow-div {
        text-align: center;
        font-size: 1.25rem;
        color: #00E676;
        line-height: 84px;
        font-weight: 700;
    }

    /* -------------------------------------------------------------------------
       METALLIC BUTTON DESIGN (Phase 15: START DETECTION & Primary Buttons)
       ------------------------------------------------------------------------- */
    div.stButton > button,
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(180deg, #14221A 0%, #0E1813 55%, #09100C 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #1B3828 !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover,
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(180deg, #1B3325 0%, #12241A 55%, #0A140E 100%) !important;
        border-color: #00E676 !important;
        color: #00E676 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 16px rgba(0, 230, 118, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.5) !important;
    }

    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, #183324 0%, #10241A 50%, #0B1912 100%) !important;
        color: #00E676 !important;
        border: 1.5px solid #00E676 !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        padding: 0.65rem 1.6rem !important;
        box-shadow: 0 2px 10px rgba(0, 230, 118, 0.15), inset 0 1px 0 rgba(0, 230, 118, 0.2) !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(180deg, #204530 0%, #153022 50%, #0D2117 100%) !important;
        border-color: #00FF85 !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 230, 118, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }
    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span,
    div.stButton > button[data-testid="stBaseButton-primary"] p,
    div.stButton > button[data-testid="stBaseButton-primary"] span {
        color: inherit !important;
        font-weight: 700 !important;
    }

    /* -------------------------------------------------------------------------
       INPUT FIELDS & CONTROLS
       ------------------------------------------------------------------------- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #080E0B !important;
        border: 1px solid #1B3828 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        font-size: 0.93rem !important;
        padding: 0.65rem 0.9rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus {
        border-color: #00E676 !important;
        box-shadow: 0 0 0 3px rgba(0, 230, 118, 0.2) !important;
        outline: none !important;
    }

    /* Selectbox styling */
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #080E0B !important;
        border: 1px solid #1B3828 !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: #00E676 !important;
        box-shadow: 0 0 0 3px rgba(0, 230, 118, 0.2) !important;
    }

    /* -------------------------------------------------------------------------
       PREDICTION RESULT CARDS (Dark Theme + Vibrant Badges)
       ------------------------------------------------------------------------- */
    .prediction-container {
        background: #0B120E;
        border: 1px solid #1B3828;
        border-radius: 12px;
        padding: 2rem 2.2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        text-align: center;
    }
    .pred-header-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #A7B5AD;
        margin-bottom: 0.75rem;
    }
    .pred-main-fake {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #EF4444;
        background: linear-gradient(180deg, #1C0B0B 0%, #120505 100%);
        border: 1.5px solid #DC2626;
        border-radius: 8px;
        padding: 0.5rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.2);
    }
    .pred-main-real {
        display: inline-block;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #00E676;
        background: linear-gradient(180deg, #0B1C12 0%, #05120A 100%);
        border: 1.5px solid #00E676;
        border-radius: 8px;
        padding: 0.5rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 0 16px rgba(0, 230, 118, 0.2);
    }
    .pred-conf-block {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.2rem;
    }
    .pred-conf-caption {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #A7B5AD;
    }
    .pred-conf-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #FFFFFF;
    }

    /* Comparison Dual-Card Layout */
    .compare-card {
        background: #0B120E;
        border: 1px solid #1B3828;
        border-radius: 11px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        text-align: center;
    }
    .compare-model-name {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #00E676;
        margin-bottom: 0.8rem;
    }

    /* Dataframe aesthetics */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1B3828 !important;
        border-radius: 8px !important;
        background-color: #080E0B !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP NAVIGATION BAR (Section 3: Tile-Style Navigation)
# -----------------------------------------------------------------------------
selected_tab = st.radio(
    "Navigation Menu",
    options=NAV_PAGES,
    key="current_nav",
    horizontal=True,
    label_visibility="collapsed"
)

# System Status Strip (Metallic Dark Bar with Live Status)
exists, _ = check_dataset_exists()
cnn_status = "Trained" if os.path.exists(config.CNN_CHECKPOINT) else "Untrained"
lstm_status = "Trained" if os.path.exists(config.LSTM_CHECKPOINT) else "Untrained"

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: #080E0B; border: 1px solid #1B3828; border-radius: 7px; padding: 0.45rem 1.2rem; margin-bottom: 1.6rem; font-size: 0.78rem; color: #A7B5AD;">
    <span><strong>System Status:</strong> Dataset: <span style="color: {'#00E676' if exists else '#EF4444'}; font-weight: 600;">{'Available (ISOT)' if exists else 'Not Found'}</span></span>
    <span>Compute: <strong style="color: #FFFFFF;">{config.DEVICE.type.upper()}</strong></span>
    <span>TextCNN: <strong style="color: {'#00E676' if cnn_status == 'Trained' else '#A7B5AD'};">{cnn_status}</strong></span>
    <span>LSTM: <strong style="color: {'#00E676' if lstm_status == 'Trained' else '#A7B5AD'};">{lstm_status}</strong></span>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: HOME
# =============================================================================
def render_home():
    st.markdown("""
    <div class="metallic-hero-card">
        <div class="hero-badge">RESEARCH PLATFORM</div>
        <div class="hero-title-main">Deep Learning-Based<br>Fake News Detection</div>
        <div class="hero-subtitle-secondary">Using TextCNN and LSTM Architectures</div>
        <div class="hero-divider"></div>
        <p class="hero-desc">
            An advanced natural language verification system that uses parallel multi-kernel 
            Convolutional Neural Networks and Long Short-Term Memory networks to classify news articles 
            with authentic empirical evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-header-title">System Workflow</div>', unsafe_allow_html=True)
    st.caption("End-to-end deep learning classification pipeline:")

    # Responsive Flowchart Nodes
    col_w1, col_a1, col_w2, col_a2, col_w3, col_a3, col_w4, col_a4, col_w5 = st.columns([2, 0.7, 2, 0.7, 2, 0.7, 2, 0.7, 2])
    with col_w1:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">News Input</div><div class="flow-step-sub">Headline + Body</div></div>', unsafe_allow_html=True)
    with col_a1:
        st.markdown('<div class="flow-arrow-div">→</div>', unsafe_allow_html=True)
    with col_w2:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">Preprocessing</div><div class="flow-step-sub">Clean & Tokenize</div></div>', unsafe_allow_html=True)
    with col_a2:
        st.markdown('<div class="flow-arrow-div">→</div>', unsafe_allow_html=True)
    with col_w3:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">CNN / LSTM</div><div class="flow-step-sub">Neural Features</div></div>', unsafe_allow_html=True)
    with col_a3:
        st.markdown('<div class="flow-arrow-div">→</div>', unsafe_allow_html=True)
    with col_w4:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">Classification</div><div class="flow-step-sub">MLP Projection</div></div>', unsafe_allow_html=True)
    with col_a4:
        st.markdown('<div class="flow-arrow-div">→</div>', unsafe_allow_html=True)
    with col_w5:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">Prediction</div><div class="flow-step-sub">Label + Confidence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cta, _ = st.columns([2.8, 6])
    with col_cta:
        if st.button("START DETECTION →", type="primary", use_container_width=True):
            navigate_to("FAKE NEWS DETECTOR")

    st.markdown("---")

    # Architecture Overview Cards
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="premium-card">
            <div class="card-header-title">Architecture: TextCNN</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">Multi-Kernel Convolutional Network</div>
            <div class="card-desc-text">
                Applies parallel 1D convolutional filter banks of sizes 3, 4, and 5 over word embeddings to detect 
                salient n-gram patterns, sensationalist rhetorical tropes, and informative local phrases.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown("""
        <div class="premium-card">
            <div class="card-header-title">Architecture: LSTM</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.4rem;">Sequential Context Network</div>
            <div class="card-desc-text">
                Processes word sequences recurrently through specialized gating mechanisms to capture 
                long-range narrative context, sequential dependencies, and stylistic consistency across the article.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: FAKE NEWS DETECTOR
# =============================================================================
def render_detector():
    st.markdown("### Fake News Detector")
    st.caption("Enter a news headline and article text to analyze with trained PyTorch deep learning models.")

    predictor = NewsPredictor()

    col_sel, _ = st.columns([3, 4])
    with col_sel:
        model_choice = st.selectbox(
            "Select Model Architecture",
            options=["Compare Both", "CNN", "LSTM"],
            index=0,
            help="Compare Both evaluates the article independently on both models for side-by-side analysis."
        )

    headline_input = st.text_input(
        "Headline",
        placeholder="Enter the news headline or article title...",
        help="Article title provides valuable n-gram framing indicators."
    )

    article_input = st.text_area(
        "Article Text",
        height=220,
        placeholder="Paste the full body of the news article here...",
        help="The full text is normalized, sanitized, tokenized, and projected into neural tensors."
    )

    col_btn, _ = st.columns([2.5, 6])
    with col_btn:
        analyze_clicked = st.button("ANALYZE NEWS", type="primary", use_container_width=True)

    if analyze_clicked:
        if not headline_input.strip() and not article_input.strip():
            st.warning("Please enter a headline or article before analyzing.")
        else:
            target_to_check = "both" if model_choice == "Compare Both" else model_choice.lower()
            if not predictor.is_model_available(target_to_check):
                st.error("Model not trained yet.\nPlease train the model before making predictions.")
                st.info("You can train the models directly from the 'TRAINING SESSION' tab in the navigation bar.")
            else:
                with st.spinner("Processing text and executing neural inference..."):
                    try:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        if model_choice == "Compare Both":
                            res = predictor.predict_both(headline_input, article_input)
                            cnn_res = res["cnn"]
                            lstm_res = res["lstm"]

                            # Record to session history
                            st.session_state.prediction_history.append({
                                "timestamp": timestamp,
                                "headline": headline_input[:60] if headline_input else article_input[:60] + "...",
                                "model": "Both (CNN + LSTM)",
                                "cnn_pred": cnn_res["prediction"],
                                "cnn_conf": f"{cnn_res['confidence']:.1f}%",
                                "lstm_pred": lstm_res["prediction"],
                                "lstm_conf": f"{lstm_res['confidence']:.1f}%",
                                "result": f"CNN: {cnn_res['prediction']} ({cnn_res['confidence']:.1f}%) | LSTM: {lstm_res['prediction']} ({lstm_res['confidence']:.1f}%)"
                            })

                            st.markdown("---")
                            st.markdown("#### Side-by-Side Model Prediction")

                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                is_fake = (cnn_res["prediction"] == "FAKE NEWS")
                                status_class = "pred-main-fake" if is_fake else "pred-main-real"
                                status_text = "FAKE" if is_fake else "REAL"
                                st.markdown(f"""
                                <div class="compare-card">
                                    <div class="compare-model-name">Convolutional Neural Network (CNN)</div>
                                    <div class="pred-header-kicker">Model Prediction</div>
                                    <div class="{status_class}">{status_text}</div>
                                    <div class="pred-conf-block">
                                        <span class="pred-conf-caption">Confidence</span>
                                        <span class="pred-conf-number">{cnn_res['confidence']:.1f}%</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            with col_m2:
                                is_fake = (lstm_res["prediction"] == "FAKE NEWS")
                                status_class = "pred-main-fake" if is_fake else "pred-main-real"
                                status_text = "FAKE" if is_fake else "REAL"
                                st.markdown(f"""
                                <div class="compare-card">
                                    <div class="compare-model-name">Long Short-Term Memory (LSTM)</div>
                                    <div class="pred-header-kicker">Model Prediction</div>
                                    <div class="{status_class}">{status_text}</div>
                                    <div class="pred-conf-block">
                                        <span class="pred-conf-caption">Confidence</span>
                                        <span class="pred-conf-number">{lstm_res['confidence']:.1f}%</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown('<div class="card-header-title">Comparison Summary</div>', unsafe_allow_html=True)
                            df_both = pd.DataFrame(res["comparison"])
                            st.table(df_both)

                        else:
                            res = predictor.predict_single(headline_input, article_input, model_type=model_choice.lower())
                            is_fake = (res["prediction"] == "FAKE NEWS")
                            status_class = "pred-main-fake" if is_fake else "pred-main-real"
                            status_text = "FAKE" if is_fake else "REAL"

                            st.session_state.prediction_history.append({
                                "timestamp": timestamp,
                                "headline": headline_input[:60] if headline_input else article_input[:60] + "...",
                                "model": res["model"],
                                "cnn_pred": res["prediction"] if res["model"] == "CNN" else "N/A",
                                "cnn_conf": f"{res['confidence']:.1f}%" if res["model"] == "CNN" else "N/A",
                                "lstm_pred": res["prediction"] if res["model"] == "LSTM" else "N/A",
                                "lstm_conf": f"{res['confidence']:.1f}%" if res["model"] == "LSTM" else "N/A",
                                "result": f"{res['prediction']} ({res['confidence']:.1f}%)"
                            })

                            st.markdown(f"""
                            <div class="prediction-container">
                                <div class="pred-header-kicker">Model Prediction ({res['model']})</div>
                                <div class="{status_class}">{status_text}</div>
                                <div class="pred-conf-block">
                                    <span class="pred-conf-caption">Confidence</span>
                                    <span class="pred-conf-number">{res['confidence']:.1f}%</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.caption(f"Architecture: {res['model']} | Evaluated Sequence Word Count: {res['word_count']} words")

                        st.caption("Note: Predictions reflect statistical and lexical patterns learned from the training corpus. The system does not independently query live web sources.")

                    except Exception as e:
                        st.error(f"Inference error: {str(e)}")


# =============================================================================
# PAGE 3: MODEL COMPARISON
# =============================================================================
def render_model_comparison():
    st.markdown("### Model Comparison & Evaluation")
    st.caption("Empirical performance metrics measured on the untouched 10% test dataset split.")

    if not os.path.exists(config.MODEL_COMPARISON_CSV):
        st.warning("Models not trained yet.\nPlease train the models before viewing performance.")
        if st.button("GO TO TRAINING SESSION", type="primary"):
            navigate_to("TRAINING SESSION")
    else:
        df_comp = pd.read_csv(config.MODEL_COMPARISON_CSV)

        if not df_comp.empty and "F1 Score" in df_comp.columns:
            best_idx = df_comp["F1 Score"].idxmax()
            best_model_name = df_comp.loc[best_idx, "Model"]
            best_f1 = df_comp.loc[best_idx, "F1 Score"]
            st.success(f"**Highest Performing Architecture:** {best_model_name} (F1 Score: {best_f1 * 100:.2f}%)")

        st.markdown('<div class="card-header-title">Test Set Performance Summary</div>', unsafe_allow_html=True)
        
        display_df = df_comp.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
        
        st.dataframe(display_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Performance Visualizations</div>', unsafe_allow_html=True)
        
        comp_plot_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
        if os.path.exists(comp_plot_path):
            st.image(comp_plot_path, caption="CNN vs LSTM Test Metric Comparison", use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        loss_acc_path = os.path.join(config.PLOTS_DIR, "loss_accuracy_curves.png")
        if os.path.exists(loss_acc_path):
            with col_p1:
                st.image(loss_acc_path, caption="Training vs Validation Learning Curves", use_container_width=True)

        cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            with col_p2:
                st.image(cm_path, caption="Confusion Matrices on Test Split", use_container_width=True)


# =============================================================================
# PAGE 4: TRAINING SESSION
# =============================================================================
def render_training_session():
    st.markdown("### Model Training Pipeline")
    st.caption("Configure hyperparameters, initiate PyTorch neural training, and monitor progress in real time.")

    exists, err_msg = check_dataset_exists()

    if not exists:
        st.error(err_msg)
        st.markdown("You can automatically download the verified ISOT dataset files using the button below:")
        if st.button("DOWNLOAD ISOT DATASET", type="primary"):
            p_bar = st.progress(0.0)
            status_t = st.empty()
            def dl_cb(pct, text):
                p_bar.progress(min(1.0, pct / 100.0))
                status_t.text(text)
            with st.spinner("Downloading ISOT dataset..."):
                ensure_dataset(progress_callback=dl_cb)
            st.success("Dataset downloaded successfully!")
            st.rerun()
    else:
        stats = get_dataset_statistics()
        if stats.get("available", False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">Total Articles</div>
                    <div class="card-val-display">{stats['total_articles']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">Fake Samples</div>
                    <div class="card-val-display" style="color: #EF4444;">{stats['fake_articles']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">Real Samples</div>
                    <div class="card-val-display" style="color: #00E676;">{stats['real_articles']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">Compute Device</div>
                    <div class="card-val-display">{config.DEVICE.type.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="card-header-title">Training Configuration</div>', unsafe_allow_html=True)

        col_m, col_info = st.columns([2.5, 3])
        with col_m:
            mode_choice = st.radio(
                "Training Mode Preset:",
                options=["quick", "normal"],
                format_func=lambda x: f"{x.capitalize()} Mode ({config.TRAINING_MODES[x]['description']})",
                index=0,
                help="Quick mode uses 5,000 real samples and 2 epochs for rapid execution."
            )

        with col_info:
            cfg = config.TRAINING_MODES[mode_choice]
            st.markdown(f"""
            <div class="premium-card" style="margin-bottom: 0;">
                <div class="card-header-title">Active Parameters</div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #A7B5AD; line-height: 1.6;">
                    <li><strong>Samples:</strong> {cfg['samples']:,} authentic articles</li>
                    <li><strong>Epochs:</strong> {cfg['epochs']}</li>
                    <li><strong>Batch Size:</strong> {cfg['batch_size']}</li>
                    <li><strong>Learning Rate:</strong> {cfg['learning_rate']}</li>
                    <li><strong>Split:</strong> 80% Train, 10% Validation, 10% Test (Stratified)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-header-title">Launch Training Session</div>', unsafe_allow_html=True)
        
        confirm_train = st.checkbox("I confirm that I want to train the selected deep learning model(s).")

        col_b1, col_b2, col_b3 = st.columns(3)
        train_cnn_btn = col_b1.button("TRAIN CNN", disabled=not confirm_train, use_container_width=True)
        train_lstm_btn = col_b2.button("TRAIN LSTM", disabled=not confirm_train, use_container_width=True)
        train_both_btn = col_b3.button("TRAIN BOTH MODELS", type="primary", disabled=not confirm_train, use_container_width=True)

        target = None
        if train_cnn_btn:
            target = "cnn"
        elif train_lstm_btn:
            target = "lstm"
        elif train_both_btn:
            target = "both"

        if target:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            log_box = st.empty()
            logs = []

            def p_cb(progress, status):
                progress_bar.progress(min(1.0, max(0.0, progress)))
                status_text.text(status)

            def l_cb(line):
                logs.append(line)
                log_box.code("\n".join(logs[-12:]))

            with st.spinner(f"Training {target.upper()} neural model(s)..."):
                try:
                    start_t = time.time()
                    results = train_pipeline(
                        target=target,
                        mode=mode_choice,
                        progress_callback=p_cb,
                        log_callback=l_cb
                    )
                    elapsed = time.time() - start_t
                    
                    with st.spinner("Evaluating models on untouched test set and rendering plots..."):
                        eval_res = evaluate_models()

                    # Save persistent training results to session state
                    st.session_state.last_training_run = {
                        "target": target.upper(),
                        "mode": mode_choice,
                        "time": f"{elapsed:.1f}s",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "logs": logs,
                        "comparison_df": eval_res["comparison_df"]
                    }

                    st.success(f"Training and evaluation finished successfully for {target.upper()} in {elapsed:.1f}s!")

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

        # Persistent Display of Latest Training Session Results (Phase 8 Fix)
        if st.session_state.last_training_run:
            run = st.session_state.last_training_run
            st.markdown("---")
            st.markdown(f"#### Latest Training Session Output ({run['target']})")
            st.caption(f"Completed at {run['timestamp']} | Mode: {run['mode'].capitalize()} | Duration: {run['time']}")
            
            with st.expander("Training Session Console Logs", expanded=False):
                st.code("\n".join(run["logs"]))

            st.markdown('<div class="card-header-title">Updated Test Evaluation Results</div>', unsafe_allow_html=True)
            display_eval = run["comparison_df"].copy()
            for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                if col in display_eval.columns:
                    display_eval[col] = display_eval[col].apply(lambda x: f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x)
            st.dataframe(display_eval, use_container_width=True)

            comp_img = os.path.join(config.PLOTS_DIR, "model_comparison.png")
            if os.path.exists(comp_img):
                st.image(comp_img, caption="Updated Model Comparison on Test Split", use_container_width=True)


# =============================================================================
# PAGE 5: SESSIONS
# =============================================================================
def render_sessions():
    st.markdown("### Sessions & Historical Analysis")
    st.caption("Inspect prediction history, model training checkpoints, and system performance sessions.")

    tab1, tab2, tab3 = st.tabs(["Prediction History", "Training Histories", "Model Checkpoints & Specs"])

    with tab1:
        st.markdown('<div class="card-header-title">Active Session Predictions</div>', unsafe_allow_html=True)
        if not st.session_state.prediction_history:
            st.info("No predictions made during this session yet. Run an analysis from the 'FAKE NEWS DETECTOR' tab to populate this history.")
        else:
            col_act, _ = st.columns([2, 6])
            with col_act:
                if st.button("CLEAR PREDICTION HISTORY", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()

            df_preds = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_preds, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card-header-title">Recent Prediction Cards</div>', unsafe_allow_html=True)
            for idx, p in enumerate(reversed(st.session_state.prediction_history[-5:])):
                with st.expander(f"[{p['timestamp']}] {p['headline']} — {p['result']}", expanded=(idx == 0)):
                    st.write(f"**Evaluated Headline / Text:** {p['headline']}")
                    st.write(f"**Model Architecture:** {p['model']}")
                    st.write(f"**CNN Result:** {p.get('cnn_pred', 'N/A')} ({p.get('cnn_conf', 'N/A')})")
                    st.write(f"**LSTM Result:** {p.get('lstm_pred', 'N/A')} ({p.get('lstm_conf', 'N/A')})")

    with tab2:
        st.markdown('<div class="card-header-title">Training Histories & Learning Curves</div>', unsafe_allow_html=True)
        
        histories_found = False
        col_h1, col_h2 = st.columns(2)

        if os.path.exists(config.CNN_HISTORY_FILE):
            histories_found = True
            with open(config.CNN_HISTORY_FILE, "r", encoding="utf-8") as f:
                h_cnn = json.load(f)
            with col_h1:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">TextCNN Training Summary</div>
                    <p style="color: #FFFFFF; font-weight: 700; margin-bottom: 0.4rem;">Best Validation Accuracy: {h_cnn.get('best_val_acc', 0)*100:.2f}%</p>
                    <ul style="color: #A7B5AD; font-size: 0.85rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                        <li><strong>Epochs:</strong> {h_cnn.get('epochs', 'N/A')}</li>
                        <li><strong>Mode:</strong> {h_cnn.get('mode', 'N/A').capitalize()}</li>
                        <li><strong>Device:</strong> {h_cnn.get('device', 'N/A')}</li>
                        <li><strong>Training Time:</strong> {h_cnn.get('total_training_time', 'N/A')}s</li>
                        <li><strong>Final Train Loss:</strong> {h_cnn['train_loss'][-1] if h_cnn.get('train_loss') else 'N/A'}</li>
                        <li><strong>Final Val Loss:</strong> {h_cnn['val_loss'][-1] if h_cnn.get('val_loss') else 'N/A'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        if os.path.exists(config.LSTM_HISTORY_FILE):
            histories_found = True
            with open(config.LSTM_HISTORY_FILE, "r", encoding="utf-8") as f:
                h_lstm = json.load(f)
            with col_h2:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">LSTM Training Summary</div>
                    <p style="color: #FFFFFF; font-weight: 700; margin-bottom: 0.4rem;">Best Validation Accuracy: {h_lstm.get('best_val_acc', 0)*100:.2f}%</p>
                    <ul style="color: #A7B5AD; font-size: 0.85rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                        <li><strong>Epochs:</strong> {h_lstm.get('epochs', 'N/A')}</li>
                        <li><strong>Mode:</strong> {h_lstm.get('mode', 'N/A').capitalize()}</li>
                        <li><strong>Device:</strong> {h_lstm.get('device', 'N/A')}</li>
                        <li><strong>Training Time:</strong> {h_lstm.get('total_training_time', 'N/A')}s</li>
                        <li><strong>Final Train Loss:</strong> {h_lstm['train_loss'][-1] if h_lstm.get('train_loss') else 'N/A'}</li>
                        <li><strong>Final Val Loss:</strong> {h_lstm['val_loss'][-1] if h_lstm.get('val_loss') else 'N/A'}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        if not histories_found:
            st.info("No saved model training history files found yet. Train models in the 'TRAINING SESSION' tab to generate training records.")
        else:
            loss_acc_img = os.path.join(config.PLOTS_DIR, "loss_accuracy_curves.png")
            if os.path.exists(loss_acc_img):
                st.image(loss_acc_img, caption="Training & Validation Curves Across Epochs", use_container_width=True)

    with tab3:
        st.markdown('<div class="card-header-title">Active Model Checkpoints & Specifications</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-header-title">TextCNN Architecture Specs</div>
                <ul style="color: #A7B5AD; font-size: 0.86rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Embedding Dimension:</strong> {config.CNN_CONFIG['embedding_dim']}</li>
                    <li><strong>Filter Banks:</strong> {config.CNN_CONFIG['num_filters']} filters per kernel</li>
                    <li><strong>Kernel Sizes:</strong> {config.CNN_CONFIG['kernel_sizes']}</li>
                    <li><strong>Classification Head:</strong> 2-Stage MLP (300 → 128 → 1) with Dropout</li>
                    <li><strong>Checkpoint Status:</strong> {'Available' if os.path.exists(config.CNN_CHECKPOINT) else 'Not Found'}</li>
                    <li><strong>Checkpoint Path:</strong> checkpoints/cnn_best.pth</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-header-title">LSTM Architecture Specs</div>
                <ul style="color: #A7B5AD; font-size: 0.86rem; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Embedding Dimension:</strong> {config.LSTM_CONFIG['embedding_dim']}</li>
                    <li><strong>Hidden Dimension:</strong> {config.LSTM_CONFIG['hidden_dim']}</li>
                    <li><strong>Recurrent Layers:</strong> {config.LSTM_CONFIG['num_layers']}</li>
                    <li><strong>Dropout:</strong> {config.LSTM_CONFIG['dropout']}</li>
                    <li><strong>Checkpoint Status:</strong> {'Available' if os.path.exists(config.LSTM_CHECKPOINT) else 'Not Found'}</li>
                    <li><strong>Checkpoint Path:</strong> checkpoints/lstm_best.pth</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# CLEAN CONDITIONAL PAGE ROUTING (Phase 5 Fix: Only Active Page Renders)
# =============================================================================
page_container = st.container()

with page_container:
    if selected_tab == "HOME":
        render_home()
    elif selected_tab == "FAKE NEWS DETECTOR":
        render_detector()
    elif selected_tab == "MODEL COMPARISON":
        render_model_comparison()
    elif selected_tab == "TRAINING SESSION":
        render_training_session()
    elif selected_tab == "SESSIONS":
        render_sessions()
