"""
TRUTHSCAN AI - Web Application
Deep Learning-Based Fake News Detection Using CNN and LSTM
Premium Light Metallic Theme UI.
"""

import os
import json
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
# Premium Light Metallic Theme CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Font: Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }

    /* Main Container Padding - Below Streamlit Header */
    .block-container {
        padding-top: 4.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }

    /* -------------------------------------------------------------------------
       TOP NAVIGATION TILES (Section 3 & 4)
       ------------------------------------------------------------------------- */
    /* Hide the radio widget label "Navigation Menu" */
    div[data-testid="stRadio"] > label[data-testid="stWidgetLabel"],
    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 0.85rem !important;
        justify-content: center !important;
        align-items: center !important;
        padding: 0.5rem 0 1.25rem 0 !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }

    /* Wrap item */
    div[data-testid="stRadioGroup"] > div {
        margin: 0 !important;
        padding: 0 !important;
        display: inline-flex !important;
    }

    /* Navigation Tile Design (Section 3) */
    div[data-testid="stRadioGroup"] label[data-testid="stRadioOption"],
    label[data-testid="stRadioOption"],
    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 55%, #f1f5f9 100%) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.35rem !important;
        cursor: pointer !important;
        box-shadow: 0 2px 4px rgba(148, 163, 184, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        min-width: 175px !important;
        margin: 0 !important;
    }

    /* Position native radio input off-screen without breaking event dispatch */
    label[data-testid="stRadioOption"] input {
        opacity: 0 !important;
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        pointer-events: none !important;
    }

    /* Completely hide native circular radio button icons and graphics */
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
        color: #475569 !important;
        text-transform: uppercase !important;
        transition: all 0.22s ease !important;
        margin: 0 !important;
        text-align: center !important;
    }

    /* TILE HOVER EFFECT (Section 4) */
    label[data-testid="stRadioOption"]:hover,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        transform: translateY(-2.5px) scale(1.018) !important;
        border-color: #94a3b8 !important;
        background: linear-gradient(180deg, #ffffff 0%, #edf2f7 50%, #e2e8f0 100%) !important;
        box-shadow: 0 6px 16px rgba(100, 116, 139, 0.14), 0 2px 4px rgba(100, 116, 139, 0.06), inset 0 1px 0 #ffffff !important;
    }
    label[data-testid="stRadioOption"]:hover p,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* ACTIVE TILE */
    label[data-testid="stRadioOption"][data-selected="true"],
    div[data-testid="stRadioGroup"] div[data-selected="true"] label[data-testid="stRadioOption"],
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(180deg, #e2e8f0 0%, #cbd5e1 55%, #94a3b8 100%) !important;
        border: 1.5px solid #334155 !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.08), 0 3px 8px rgba(100, 116, 139, 0.15) !important;
        transform: translateY(0) scale(1.0) !important;
    }
    label[data-testid="stRadioOption"][data-selected="true"] p,
    div[data-testid="stRadioGroup"] div[data-selected="true"] label[data-testid="stRadioOption"] p,
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* -------------------------------------------------------------------------
       HERO CARD (Section 2 & 5)
       ------------------------------------------------------------------------- */
    .metallic-hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 45%, #f1f5f9 85%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 2.6rem 2.8rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(148, 163, 184, 0.1), inset 0 1px 0 #ffffff;
        position: relative;
    }
    .metallic-hero-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #94a3b8 0%, #cbd5e1 50%, #94a3b8 100%);
        border-radius: 14px 14px 0 0;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.7rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    .hero-title-main {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.15;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle-primary {
        font-size: 1.25rem;
        font-weight: 600;
        color: #334155;
        margin: 0 0 0.35rem 0;
    }
    .hero-subtitle-secondary {
        font-size: 1rem;
        font-weight: 500;
        color: #64748b;
        margin: 0 0 1rem 0;
    }
    .hero-divider {
        width: 50px;
        height: 2px;
        background: #94a3b8;
        margin: 1rem 0;
    }
    .hero-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
        max-width: 820px;
        margin: 0;
    }

    /* -------------------------------------------------------------------------
       CLEAN CARDS (Section 7)
       ------------------------------------------------------------------------- */
    .premium-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 8px rgba(148, 163, 184, 0.06), inset 0 1px 0 #ffffff;
    }
    .card-header-title {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .card-val-display {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    .card-desc-text {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.55;
        margin-top: 0.4rem;
    }

    /* Flowchart Tile Nodes */
    .flow-step-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #cbd5e1;
        border-radius: 9px;
        padding: 0.85rem 0.6rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(148, 163, 184, 0.05);
        min-height: 84px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .flow-step-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.15rem;
    }
    .flow-step-sub {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 500;
    }
    .flow-arrow-div {
        text-align: center;
        font-size: 1.2rem;
        color: #94a3b8;
        line-height: 84px;
        font-weight: 600;
    }

    /* -------------------------------------------------------------------------
       BUTTON DESIGN (Section 6)
       Metallic silver appearance, rounded corners, dark text, thin border,
       subtle shadow, smooth hover effect
       ------------------------------------------------------------------------- */
    div.stButton > button,
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 55%, #e2e8f0 100%) !important;
        color: #0f172a !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 2px 5px rgba(148, 163, 184, 0.14), inset 0 1px 0 #ffffff !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover,
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(180deg, #ffffff 0%, #e2e8f0 55%, #cbd5e1 100%) !important;
        border-color: #64748b !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 14px rgba(100, 116, 139, 0.2), inset 0 1px 0 #ffffff !important;
        color: #020617 !important;
    }
    div.stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.08) !important;
    }

    /* Primary Button Style (Section 6: Metallic Silver / Brushed Platinum with Dark Text) */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%) !important;
        color: #0f172a !important;
        border: 1.5px solid #64748b !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em !important;
        padding: 0.65rem 1.6rem !important;
        box-shadow: 0 2px 6px rgba(100, 116, 139, 0.16), inset 0 1px 0 #ffffff !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%) !important;
        border-color: #334155 !important;
        color: #020617 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(71, 85, 105, 0.25), inset 0 1px 0 #ffffff !important;
    }
    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] span,
    div.stButton > button[data-testid="stBaseButton-primary"] p,
    div.stButton > button[data-testid="stBaseButton-primary"] span {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* -------------------------------------------------------------------------
       INPUT FIELDS (Section 11)
       ------------------------------------------------------------------------- */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #0f172a !important;
        font-size: 0.93rem !important;
        padding: 0.65rem 0.9rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.25) !important;
        outline: none !important;
    }

    /* Selectbox styling */
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: #64748b !important;
        box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.25) !important;
    }

    /* -------------------------------------------------------------------------
       PREDICTION RESULT DISPLAY (Section 8)
       ------------------------------------------------------------------------- */
    .prediction-container {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 2rem 2.2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(148, 163, 184, 0.08), inset 0 1px 0 #ffffff;
        text-align: center;
    }
    .pred-header-kicker {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.75rem;
    }
    .pred-main-fake {
        display: inline-block;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #b91c1c;
        background: linear-gradient(180deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #f87171;
        border-radius: 8px;
        padding: 0.5rem 2.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.1);
    }
    .pred-main-real {
        display: inline-block;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #065f46;
        background: linear-gradient(180deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1.5px solid #34d399;
        border-radius: 8px;
        padding: 0.5rem 2.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.1);
    }
    .pred-conf-block {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.2rem;
    }
    .pred-conf-caption {
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748b;
    }
    .pred-conf-number {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }

    /* Comparison Dual-Card Layout */
    .compare-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 11px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 3px 10px rgba(148, 163, 184, 0.06), inset 0 1px 0 #ffffff;
        text-align: center;
    }
    .compare-model-name {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.8rem;
    }

    /* Dataframe & Table Aesthetics */
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP NAVIGATION BAR (Section 3: Tile-Style Navigation)
# -----------------------------------------------------------------------------
NAV_OPTIONS = [
    "HOME",
    "FAKE NEWS DETECTOR",
    "MODEL COMPARISON",
    "TRAINING",
    "ABOUT"
]

if "current_nav" not in st.session_state:
    st.session_state.current_nav = "HOME"

selected_tab = st.radio(
    "Navigation Menu",
    options=NAV_OPTIONS,
    key="current_nav",
    horizontal=True,
    label_visibility="collapsed"
)

# System Status Sub-header Bar (Clean Metallic Status Strip)
exists, _ = check_dataset_exists()
cnn_status = "Trained" if os.path.exists(config.CNN_CHECKPOINT) else "Untrained"
lstm_status = "Trained" if os.path.exists(config.LSTM_CHECKPOINT) else "Untrained"

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 7px; padding: 0.45rem 1.2rem; margin-bottom: 1.6rem; font-size: 0.78rem; color: #64748b;">
    <span><strong>System Status:</strong> ISOT Dataset: <span style="color: {'#059669' if exists else '#dc2626'}; font-weight: 600;">{'Available' if exists else 'Not Found'}</span></span>
    <span>Compute: <strong style="color: #334155;">{config.DEVICE.type.upper()}</strong></span>
    <span>TextCNN: <strong style="color: {'#059669' if cnn_status == 'Trained' else '#64748b'};">{cnn_status}</strong></span>
    <span>LSTM: <strong style="color: {'#059669' if lstm_status == 'Trained' else '#64748b'};">{lstm_status}</strong></span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 1: HOME
# -----------------------------------------------------------------------------
if selected_tab == "HOME":
    st.markdown("""
    <div class="metallic-hero-card">
        <div class="hero-brand">TRUTHSCAN AI</div>
        <div class="hero-title-main">Deep Learning-Based<br>Fake News Detection</div>
        <div class="hero-subtitle-secondary">Using CNN and LSTM</div>
        <div class="hero-divider"></div>
        <p class="hero-desc">
            An AI-based natural language verification system that uses Convolutional Neural Networks and 
            Long Short-Term Memory networks to classify news articles as real or fake.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-header-title">System Workflow</div>', unsafe_allow_html=True)
    st.caption("End-to-end neural classification pipeline:")

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
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">Classification</div><div class="flow-step-sub">Dense + Sigmoid</div></div>', unsafe_allow_html=True)
    with col_a4:
        st.markdown('<div class="flow-arrow-div">→</div>', unsafe_allow_html=True)
    with col_w5:
        st.markdown('<div class="flow-step-card"><div class="flow-step-title">Prediction</div><div class="flow-step-sub">Label + Confidence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cta, _ = st.columns([2.5, 6])
    with col_cta:
        if st.button("START DETECTION →", type="primary", use_container_width=True):
            st.session_state.current_nav = "FAKE NEWS DETECTOR"
            st.rerun()

    st.markdown("---")

    # Architecture Overview Cards
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="premium-card">
            <div class="card-header-title">Architecture: TextCNN</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">Multi-Kernel Convolutional Network</div>
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
            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">Sequential Context Network</div>
            <div class="card-desc-text">
                Processes word sequences recurrently through specialized gating mechanisms to capture 
                long-range context, narrative progression, and semantic dependencies across the article.
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: FAKE NEWS DETECTOR
# -----------------------------------------------------------------------------
elif selected_tab == "FAKE NEWS DETECTOR":
    st.markdown("### Fake News Detector")
    st.caption("Enter a news headline and article text to analyze with trained deep learning models.")

    predictor = NewsPredictor()

    col_sel, _ = st.columns([3, 4])
    with col_sel:
        model_choice = st.selectbox(
            "Select Model Architecture",
            options=["Compare Both", "CNN", "LSTM"],
            index=0,
            help="Compare Both evaluates the article independently on both models for side-by-side analysis."
        )

    # Input Fields (Section 11)
    headline_input = st.text_input(
        "Headline",
        placeholder="Enter the news headline or article title...",
        help="Article title provides valuable n-gram framing indicators."
    )

    article_input = st.text_area(
        "Article Text",
        height=220,
        placeholder="Paste the full body of the news article here...",
        help="The full text is normalized, tokenized, and encoded into sequential tensors."
    )

    col_btn, _ = st.columns([2.5, 6])
    with col_btn:
        analyze_clicked = st.button("ANALYZE NEWS", type="primary", use_container_width=True)

    if analyze_clicked:
        # Input Validation
        if not headline_input.strip() and not article_input.strip():
            st.warning("Please enter a headline or article before analyzing.")
        else:
            target_to_check = "both" if model_choice == "Compare Both" else model_choice.lower()
            if not predictor.is_model_available(target_to_check):
                st.error("Model not trained yet.\nPlease train the model before making predictions.")
                st.info("You can train the models directly from the 'TRAINING' tab in the navigation bar.")
            else:
                with st.spinner("Processing text and executing neural inference..."):
                    try:
                        if model_choice == "Compare Both":
                            res = predictor.predict_both(headline_input, article_input)
                            cnn_res = res["cnn"]
                            lstm_res = res["lstm"]

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


# -----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif selected_tab == "MODEL COMPARISON":
    st.markdown("### Model Comparison & Evaluation")
    st.caption("Empirical performance metrics measured on the untouched 10% test dataset split.")

    if not os.path.exists(config.MODEL_COMPARISON_CSV):
        st.warning("Models not trained yet.\nPlease train the models before viewing performance.")
        st.info("Navigate to the **TRAINING** tab to train the CNN and LSTM models.")
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


# -----------------------------------------------------------------------------
# TAB 4: TRAINING
# -----------------------------------------------------------------------------
elif selected_tab == "TRAINING":
    st.markdown("### Model Training Pipeline")
    st.caption("Configure hyperparameters and train PyTorch deep learning models.")

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
        # Dataset Overview Cards (Section 7)
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
                    <div class="card-val-display" style="color: #b91c1c;">{stats['fake_articles']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="premium-card">
                    <div class="card-header-title">Real Samples</div>
                    <div class="card-val-display" style="color: #065f46;">{stats['real_articles']:,}</div>
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
                help="Quick mode uses 5,000 real samples and 2 epochs for rapid execution on standard laptops."
            )

        with col_info:
            cfg = config.TRAINING_MODES[mode_choice]
            st.markdown(f"""
            <div class="premium-card" style="margin-bottom: 0;">
                <div class="card-header-title">Active Parameters</div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
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
                    results = train_pipeline(
                        target=target,
                        mode=mode_choice,
                        progress_callback=p_cb,
                        log_callback=l_cb
                    )
                    
                    st.success(f"Training finished successfully for {target.upper()}!")
                    
                    with st.spinner("Evaluating models on test set and rendering plots..."):
                        eval_res = evaluate_models()
                    
                    st.info("Evaluation results and charts updated in Model Comparison tab.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 5: ABOUT
# -----------------------------------------------------------------------------
elif selected_tab == "ABOUT":
    st.markdown("### About TRUTHSCAN AI")
    st.caption("Deep learning research project for natural language veracity classification.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="premium-card">
            <div class="card-header-title">Project Overview</div>
            <p class="card-desc-text">
                <strong>TRUTHSCAN AI</strong> is an automated veracity classification system built to demonstrate 
                the capabilities of deep learning architectures—specifically <strong>TextCNN</strong> and 
                <strong>LSTM</strong>—in identifying deceptive and unverified journalistic texts.
            </p>
        </div>
        <div class="premium-card">
            <div class="card-header-title">Problem Statement</div>
            <p class="card-desc-text">
                The exponential expansion of digital media has accelerated the spread of misinformation. 
                Manual fact-checking cannot scale with online publication rates, necessitating automated, 
                high-precision natural language processing classifiers.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="premium-card">
            <div class="card-header-title">Algorithms Used</div>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <li><strong>Convolutional Neural Network (TextCNN):</strong> Detects important local patterns, phrases, and n-grams in text via parallel 1D filter banks (kernels 3, 4, 5).</li>
                <li><strong>Long Short-Term Memory (LSTM):</strong> Processes sequential word order and learns dependencies between earlier and later parts of the text via recurrent gating.</li>
            </ul>
        </div>
        <div class="premium-card">
            <div class="card-header-title">Dataset</div>
            <p class="card-desc-text">
                Trained on the benchmark <strong>ISOT Fake and Real News Dataset</strong> (University of Victoria), 
                comprising verified Reuters articles (Real) and flagged unverified sources (Fake).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card">
        <div class="card-header-title">System Workflow</div>
        <pre style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.8rem; font-size: 0.84rem; color: #334155; margin-top: 0.5rem;">
Dataset (Fake.csv + True.csv)  →  Text Preprocessing  →  Tokenizer & Vocabulary (Train Split Only)
                                         ↓
                                CNN / LSTM Training (BCEWithLogitsLoss + Adam)
                                         ↓
                                Test Set Evaluation (Acc, Prec, Rec, F1)
                                         ↓
                                Veracity Prediction + Calibrated Confidence Score
        </pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="premium-card">
        <div class="card-header-title">Evaluation Metrics & Limitations</div>
        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; color: #334155; line-height: 1.6;">
            <li><strong>Accuracy:</strong> Proportion of total articles correctly identified as real or fake.</li>
            <li><strong>Precision & Recall:</strong> Exactitude and completeness in identifying deceptive articles.</li>
            <li><strong>F1-Score:</strong> Harmonic mean balancing false positives and false negatives.</li>
            <li><strong>Academic Limitation:</strong> Predictions are based on structural and linguistic patterns learned from the ISOT dataset and do not independently verify external breaking facts.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
