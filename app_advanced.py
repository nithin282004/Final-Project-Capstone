import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from statistics import NormalDist
import json
import joblib
import os
import re
from html import escape
import openai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# Optional hardcoded credentials (not recommended for shared/public repos).
OPENAI_API_KEY_HARDCODED = "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"
OPENAI_MODEL_HARDCODED = "gpt-4o-mini"

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Comparative Analysis of Deep Learning and Machine Learning Models for CO2 Forecasting",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional, modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --ink-900: #0f172a;
        --ink-700: #334155;
        --ink-500: #64748b;
        --sky-600: #0284c7;
        --sky-500: #0ea5e9;
        --teal-400: #2dd4bf;
        --panel: rgba(255, 255, 255, 0.9);
        --panel-border: rgba(148, 163, 184, 0.28);
    }

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(80rem 50rem at 6% -5%, rgba(2, 132, 199, 0.16), transparent 55%),
            radial-gradient(70rem 40rem at 95% 0%, rgba(45, 212, 191, 0.13), transparent 60%),
            linear-gradient(180deg, #f7fbff 0%, #eef5fb 55%, #f8fbff 100%);
        color: var(--ink-900);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b2947 0%, #12345a 58%, #0f2f52 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }

    section[data-testid="stSidebar"] * {
        color: #e2edf8 !important;
    }

    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label {
        color: #e2edf8 !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #0f172a !important;
    }

    /* Main Header */
    .main-header {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: #f8fafc !important;
        text-align: center;
        padding: 48px 22px;
        border-radius: 18px;
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.28);
        margin: 0 auto 20px auto;
        letter-spacing: 0.01em;
        text-shadow: 0 2px 10px rgba(15, 23, 42, 0.45);
        font-size: 1rem;
        line-height: 1.6;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .main-header * {
        color: #f8fafc !important;
    }

    .subtitle {
        color: #35506a;
        font-size: 1.05rem;
        text-align: center;
        margin: 0 0 26px 0;
        font-weight: 500;
    }

    .section-header {
        color: #0a4f83;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 34px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(14, 165, 233, 0.35);
    }

    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e6fffb 100%);
        padding: 18px;
        border: 1px solid rgba(14, 165, 233, 0.28);
        border-left: 5px solid var(--sky-500);
        border-radius: 12px;
        margin: 14px 0;
        box-shadow: 0 8px 26px rgba(14, 165, 233, 0.16);
        color: var(--ink-900);
    }

    .success-box {
        background: linear-gradient(135deg, #edfff7 0%, #eafff0 100%);
        padding: 18px;
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        margin: 14px 0;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.14);
        color: var(--ink-900);
    }

    .warning-box {
        background: linear-gradient(135deg, #fff9eb 0%, #fffef2 100%);
        padding: 18px;
        border: 1px solid rgba(251, 146, 60, 0.28);
        border-left: 5px solid #f97316;
        border-radius: 12px;
        margin: 14px 0;
        color: var(--ink-900);
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 14px;
        padding: 14px 12px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .metric-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-left: 5px solid #0284c7;
        border-radius: 14px;
        padding: 12px 14px;
        margin: 6px 0;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .metric-card .metric-label {
        font-size: 0.82rem;
        color: var(--ink-500);
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 6px;
    }

    .metric-card .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
        color: var(--ink-900);
    }

    .metric-card .metric-subtext {
        font-size: 0.78rem;
        color: var(--ink-500);
        margin-top: 4px;
        font-weight: 600;
    }

    .metric-tone-success {
        border-left-color: #16a34a;
    }

    .metric-tone-success .metric-value {
        color: #15803d;
    }

    .metric-tone-info {
        border-left-color: #0284c7;
    }

    .metric-tone-info .metric-value {
        color: #0369a1;
    }

    .metric-tone-warning {
        border-left-color: #ea580c;
    }

    .metric-tone-warning .metric-value {
        color: #c2410c;
    }

    .metric-tone-danger {
        border-left-color: #dc2626;
    }

    .metric-tone-danger .metric-value {
        color: #b91c1c;
    }

    .metric-tone-neutral {
        border-left-color: #475569;
    }

    .metric-tone-neutral .metric-value {
        color: #334155;
    }

    div[data-baseweb="select"], div[data-baseweb="input"] {
        border-radius: 12px;
    }

    label {
        color: var(--ink-900) !important;
        font-weight: 600 !important;
    }

    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label {
        color: var(--ink-900) !important;
        font-weight: 600 !important;
        display: block !important;
        margin-bottom: 8px !important;
    }

    .stNumberInput > div > div > label,
    .stSelectbox > div > div > label,
    .stTextInput > div > div > label {
        color: var(--ink-900) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #f8fbff !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(15, 23, 42, 0.22);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
        color: #f8fafc !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f4c81 0%, #0369a1 100%) !important;
        border: 1px solid rgba(3, 105, 161, 0.45) !important;
    }

    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
    }

    .stFormSubmitButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(3, 105, 161, 0.45) !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
        color: #f8fafc !important;
        background: linear-gradient(135deg, #0f4c81 0%, #0369a1 100%) !important;
    }

    .stFormSubmitButton > button p,
    .stFormSubmitButton > button span {
        color: #f8fafc !important;
    }

    .stButton > button p,
    .stButton > button span {
        color: #f8fafc !important;
    }

    div[data-testid="stAlert"],
    div[data-testid="stAlert"] * {
        color: var(--ink-900) !important;
    }

    /* Tabs: ensure labels are always readable */
    div[data-baseweb="tab-list"] button {
        background: rgba(226, 232, 240, 0.55) !important;
        border: 1px solid rgba(148, 163, 184, 0.45) !important;
        border-radius: 10px 10px 0 0 !important;
    }

    div[data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #ffffff !important;
        border-bottom: 2px solid #0ea5e9 !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #0369a1 !important;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #5b7088;
        font-size: 0.9em;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid rgba(148, 163, 184, 0.32);
        font-weight: 500;
    }
    
    h3 {
        color: #0f4c81;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Comparative Analysis of Deep Learning for <b>Carbon Emission Forecasting</b></h1>", unsafe_allow_html=True)

BASE_FEATURE_ORDER = [
    'population', 'gdp', 'coal_co2', 'oil_co2',
    'gas_co2', 'methane', 'nitrous_oxide', 'primary_energy_consumption'
]

MODEL_FEATURE_ORDER = ['country_encoded', 'year'] + BASE_FEATURE_ORDER
YEAR_PROJECTION_GROWTH = 0.06
YEAR_IMPACT_MULTIPLIER = 2.0

# ===========================
# LOAD MODELS & DATA
# ===========================

@st.cache_resource
def load_regression_models(version_token=None):
    model_files = {
        'random_forest': 'models/random_forest_model.pkl',
        'xgboost': 'models/xgboost_model.pkl'
    }

    loaded_models = {}
    load_errors = []

    for model_key, model_path in model_files.items():
        if not os.path.exists(model_path):
            continue
        try:
            loaded_models[model_key] = joblib.load(model_path)
        except Exception as e:
            load_errors.append(f"Failed to load {model_key} from {model_path}: {e}")

    return loaded_models, load_errors

@st.cache_resource
def load_deep_learning_model(model_key):
    model_files = {
        'lstm': 'models/lstm_model.h5',
        'rnn': 'models/rnn_model.h5'
    }

    if model_key not in model_files:
        return None, f"Unknown deep learning model key: {model_key}"

    model_path = model_files[model_key]
    if not os.path.exists(model_path):
        return None, f"Deep learning model file not found: {model_path}"

    class DenseCompat(Dense):
        """Compatibility Dense layer for models saved with extra config keys."""

        @classmethod
        def from_config(cls, config):
            cfg = dict(config)
            cfg.pop('quantization_config', None)
            return super().from_config(cfg)

    try:
        return load_model(model_path, compile=False), None
    except Exception:
        try:
            return load_model(
                model_path,
                compile=False,
                custom_objects={'Dense': DenseCompat}
            ), None
        except Exception as compat_error:
            return None, f"Failed to load {model_key} from {model_path}: {compat_error}"

def get_available_deep_model_keys():
    model_files = {
        'lstm': 'models/lstm_model.h5',
        'rnn': 'models/rnn_model.h5'
    }
    return [key for key, path in model_files.items() if os.path.exists(path)]

def get_file_version_token(path):
    """Return a simple version token so cached loaders refresh when files change."""
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

@st.cache_resource
def load_scaler(version_token=None):
    try:
        return joblib.load('models/scaler_regression.pkl'), None
    except Exception as e:
        return None, f"Failed to load scaler from models/scaler_regression.pkl: {e}"

@st.cache_resource
def load_feature_info():
    try:
        with open('models/feature_info.json', 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load feature info from models/feature_info.json: {e}"

@st.cache_resource
def load_metadata():
    try:
        with open('models/metadata.json', 'r') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load metadata from models/metadata.json: {e}"

@st.cache_data
def load_country_profiles(version_token=None):
    try:
        df = pd.read_csv('owid-co2-data-rows-dropped.backup-before-web-refresh.csv', usecols=['country', 'year', 'co2'] + BASE_FEATURE_ORDER)
        numeric_cols = ['year', 'co2'] + BASE_FEATURE_ORDER
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        trained_countries = set()
        if os.path.exists('models/metadata.json'):
            with open('models/metadata.json', 'r') as f:
                metadata = json.load(f)
            trained_countries = set(metadata.get('countries_trained', []))

        df = df.dropna(subset=BASE_FEATURE_ORDER)
        if df.empty:
            return {}, [], "No country profiles contain all required features."

        latest_profiles = df.sort_values(['country', 'year']).groupby('country', as_index=False).tail(1)

        profiles = {}
        for _, row in latest_profiles.iterrows():
            profiles[row['country']] = {
                'year': int(row['year']),
                'features': {feature: float(row[feature]) for feature in BASE_FEATURE_ORDER},
                'co2': float(row['co2']) if pd.notna(row['co2']) else None,
                'is_trained': row['country'] in trained_countries
            }

        countries = sorted(profiles.keys())
        return profiles, countries, None
    except Exception as e:
        return {}, [], f"Failed to load country profiles from owid-co2-data-rows-dropped.backup-before-web-refresh.csv: {e}"

@st.cache_resource
def load_all_performance(version_token=None):
    try:
        return pd.read_csv('models/all_models_performance.csv'), None
    except Exception as e:
        return None, f"Failed to load model performance from models/all_models_performance.csv: {e}"

# Load resources
regression_models, regression_load_errors = load_regression_models(
    (
        get_file_version_token('models/random_forest_model.pkl'),
        get_file_version_token('models/xgboost_model.pkl')
    )
)
dl_models = {}
available_deep_model_keys = get_available_deep_model_keys()
scaler, scaler_load_error = load_scaler(get_file_version_token('models/scaler_regression.pkl'))
feature_info, feature_info_error = load_feature_info()
metadata, metadata_error = load_metadata()
country_profiles, available_countries, country_profiles_error = load_country_profiles(
    get_file_version_token('owid-co2-data-rows-dropped.backup-before-web-refresh.csv')
)
all_perf_df, all_perf_error = load_all_performance(
    get_file_version_token('models/all_models_performance.csv')
)

load_issues = []
if regression_load_errors:
    load_issues.extend(regression_load_errors)
if scaler_load_error:
    load_issues.append(scaler_load_error)
if feature_info_error:
    load_issues.append(feature_info_error)
if metadata_error:
    load_issues.append(metadata_error)
if country_profiles_error:
    load_issues.append(country_profiles_error)
if all_perf_error:
    load_issues.append(all_perf_error)

if 'lstm' not in available_deep_model_keys:
    st.error("LSTM model not found! Please check models/lstm_model.h5.")
    st.stop()

if load_issues:
    st.caption("Some optional resources are unavailable. Core prediction features remain active.")
    with st.expander("Show loading details"):
        for issue in load_issues:
            st.write(f"- {issue}")

# ===========================
# HELPER FUNCTIONS
# ===========================

FEATURE_ORDER = BASE_FEATURE_ORDER.copy()

def get_country_encoded_value(country_name, info_obj):
    """Map country name to encoded numeric value expected by the model."""
    if not info_obj:
        return 0.0
    encoding_map = info_obj.get('country_encoding', {})
    if not isinstance(encoding_map, dict) or not encoding_map:
        return 0.0
    if country_name in encoding_map:
        return float(encoding_map[country_name])
    return float(np.mean(list(encoding_map.values())))

def build_model_features(features_obj, selected_country_name, info_obj):
    """Build full model feature payload in the trained feature order."""
    features_data = features_obj or {}
    return {
        'country_encoded': get_country_encoded_value(selected_country_name, info_obj),
        'year': float(features_data.get('year', 2022)),
        'population': float(features_data.get('population', 0.0)),
        'gdp': float(features_data.get('gdp', 0.0)),
        'coal_co2': float(features_data.get('coal_co2', 0.0)),
        'oil_co2': float(features_data.get('oil_co2', 0.0)),
        'gas_co2': float(features_data.get('gas_co2', 0.0)),
        'methane': float(features_data.get('methane', 0.0)),
        'nitrous_oxide': float(features_data.get('nitrous_oxide', 0.0)),
        'primary_energy_consumption': float(features_data.get('primary_energy_consumption', 0.0))
    }

def apply_feature_weighting(X_scaled, info_obj):
    """Apply optional post-scaling feature weighting (e.g., year emphasis)."""
    if X_scaled is None or info_obj is None:
        return X_scaled
    try:
        feature_cols = info_obj.get('feature_columns', [])
        year_weight = float(info_obj.get('year_weight', 1.0))
        if year_weight == 1.0 or 'year' not in feature_cols:
            return X_scaled
        year_idx = feature_cols.index('year')
        X_out = np.array(X_scaled, copy=True)
        X_out[:, year_idx] = X_out[:, year_idx] * year_weight
        return X_out
    except Exception:
        return X_scaled

def auto_adjust_inputs(previous_features, input_features):
    """Auto-adjust key scenario inputs when year/population changes.

    Only fields that the user did not manually edit are auto-adjusted.
    """
    if not previous_features:
        return input_features, []

    prev_year = int(previous_features.get('year', input_features.get('year', 2022)))
    new_year = int(input_features.get('year', prev_year))
    year_delta = new_year - prev_year

    prev_pop = max(float(previous_features.get('population', input_features.get('population', 1.0))), 1.0)
    input_pop = max(float(input_features.get('population', prev_pop)), 1.0)

    adjusted = dict(input_features)
    adjusted_fields = []

    # If year changed and population was not manually modified, project population too.
    pop_tolerance = max(abs(prev_pop) * 1e-6, 1e-8)
    population_manually_changed = abs(input_pop - prev_pop) > pop_tolerance
    if (not population_manually_changed) and year_delta != 0:
        population_annual_growth = 0.018
        projected_pop = max(prev_pop * ((1 + population_annual_growth) ** year_delta), 1.0)
        adjusted['population'] = float(projected_pop)
        if abs(projected_pop - prev_pop) > pop_tolerance:
            adjusted_fields.append('population')

    new_pop = max(float(adjusted.get('population', prev_pop)), 1.0)
    pop_ratio = new_pop / prev_pop

    growth_rules = {
        'gdp': (0.038, 1.08),
        'primary_energy_consumption': (0.026, 0.98),
        'coal_co2': (0.02, 0.9),
        'oil_co2': (0.022, 0.9),
        'gas_co2': (0.025, 0.9),
        'methane': (0.015, 0.72),
        'nitrous_oxide': (0.013, 0.72)
    }

    for field, (annual_growth, pop_elasticity) in growth_rules.items():
        prev_val = float(previous_features.get(field, input_features.get(field, 0.0)))
        current_input_val = float(input_features.get(field, prev_val))

        tolerance = max(abs(prev_val) * 1e-6, 1e-8)
        manually_changed = abs(current_input_val - prev_val) > tolerance
        if manually_changed:
            continue

        factor = ((1 + annual_growth) ** year_delta) * (pop_ratio ** pop_elasticity)
        new_val = max(prev_val * factor, 0.0)
        adjusted[field] = float(new_val)
        if abs(new_val - prev_val) > tolerance:
            adjusted_fields.append(field)

    return adjusted, adjusted_fields

MODEL_LABELS = {
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost',
    'lstm': 'LSTM',
    'rnn': 'RNN'
}

MODEL_SHORT_NAMES = {
    'random_forest': 'RF',
    'xgboost': 'XGB',
    'lstm': 'LSTM',
    'rnn': 'RNN'
}

DEFAULT_FEATURES = {
    'year': 2022,
    'population': 100000000,
    'gdp': 2500000000000,
    'coal_co2': 500,
    'oil_co2': 300,
    'gas_co2': 200,
    'methane': 50,
    'nitrous_oxide': 30,
    'primary_energy_consumption': 100
}

PREDICTION_STATE_DEFAULTS = {
    'last_prediction_mean': None,
    'last_selected_model_key': None,
    'advisor_text': "",
    'advisor_source': "",
    'advisor_error': "",
    'advisor_chat_history': [],
    'last_predictions': None,
    'last_uncertainty': None,
    'last_prediction_errors': [],
    'auto_adjust_notice': ""
}

def clear_prediction_state():
    for key, default_value in PREDICTION_STATE_DEFAULTS.items():
        if isinstance(default_value, list):
            st.session_state[key] = default_value.copy()
        else:
            st.session_state[key] = default_value

def apply_country_profile(country_name):
    profile = country_profiles.get(country_name)
    if not profile:
        return

    st.session_state.features = profile['features'].copy()
    st.session_state.features['year'] = int(profile.get('year', st.session_state.features.get('year', 2022)))
    st.session_state.active_country = country_name
    st.session_state.active_country_year = profile.get('year')
    st.session_state.active_country_actual_co2 = profile.get('co2')
    st.session_state.active_country_trained = profile.get('is_trained', False)
    clear_prediction_state()

def handle_country_change():
    selected_country = st.session_state.get('selected_country')
    if selected_country == 'Custom':
        st.session_state.active_country = None
        st.session_state.active_country_year = None
        st.session_state.active_country_actual_co2 = None
        st.session_state.active_country_trained = False
        clear_prediction_state()
        return

    apply_country_profile(selected_country)

def calculate_prediction_uncertainty(predictions_list, confidence_level=0.95):
    """Calculate configurable confidence intervals for predictions."""
    predictions_array = np.array(predictions_list)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)
    confidence_level = min(max(float(confidence_level), 0.5), 0.999)
    z_score = NormalDist().inv_cdf((1 + confidence_level) / 2)
    return {
        'mean': mean,
        'std': std,
        'confidence_level': confidence_level,
        'ci_lower': mean - z_score * std,
        'ci_upper': mean + z_score * std,
        'ci_95_lower': mean - 1.96 * std,
        'ci_95_upper': mean + 1.96 * std,
        'cv': (std / mean * 100) if mean > 0 else 0
    }

def get_model_priority_order(perf_df):
    """Return model short names ordered by observed R2 (high to low)."""
    if perf_df is None or perf_df.empty or 'Model' not in perf_df.columns or 'R² Score' not in perf_df.columns:
        return []

    model_to_short = {
        'Random Forest': 'RF',
        'XGBoost': 'XGB',
        'LSTM': 'LSTM'
    }

    ordered_models = perf_df.sort_values('R² Score', ascending=False)['Model'].tolist()
    priority = []
    for model_name in ordered_models:
        short_name = model_to_short.get(model_name)
        if short_name and short_name not in priority:
            priority.append(short_name)
    return priority

def get_selected_model_performance(model_key, perf_df):
    """Return validation metrics for a selected model from all_models_performance.csv."""
    if perf_df is None or perf_df.empty or not model_key:
        return None

    perf_name_map = {
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lstm': 'LSTM',
        'rnn': 'RNN'
    }

    model_name = perf_name_map.get(model_key)
    if not model_name or 'Model' not in perf_df.columns:
        return None

    matches = perf_df[perf_df['Model'].astype(str).str.lower() == model_name.lower()]
    if matches.empty:
        return None

    row = matches.iloc[0]
    metrics = {
        'r2': float(row['R² Score']) if 'R² Score' in row and pd.notna(row['R² Score']) else None,
        'mae': float(row['MAE']) if 'MAE' in row and pd.notna(row['MAE']) else None,
        'rmse': float(row['RMSE']) if 'RMSE' in row and pd.notna(row['RMSE']) else None
    }
    return metrics

def aggregate_predictions(predictions, method='Mean', trim_fraction=0.1):
    """Aggregate multiple model predictions using expert-configurable strategies."""
    values = np.array(list(predictions.values()), dtype=float)
    if len(values) == 0:
        return None

    method_l = (method or 'Mean').lower()
    if method_l == 'median':
        return float(np.median(values))
    if method_l == 'trimmed mean':
        if len(values) < 3:
            return float(np.mean(values))
        trim_fraction = min(max(float(trim_fraction), 0.0), 0.4)
        trim_n = int(len(values) * trim_fraction)
        if trim_n == 0 or len(values) - (2 * trim_n) <= 0:
            return float(np.mean(values))
        sorted_values = np.sort(values)
        return float(np.mean(sorted_values[trim_n:len(values)-trim_n]))
    return float(np.mean(values))

def get_available_model_keys(reg_models, deep_model_keys):
    """Return available model keys in display order."""
    ordered = ['xgboost', 'random_forest', 'lstm', 'rnn']
    deep_keys = set(deep_model_keys or [])
    available = []
    for model_key in ordered:
        if model_key in reg_models or model_key in deep_keys:
            available.append(model_key)
    return available

def get_best_available_model_key(perf_df, available_model_keys):
    """Return best available model key by highest validation R2."""
    if not available_model_keys:
        return None
    if perf_df is None or perf_df.empty or 'Model' not in perf_df.columns or 'R² Score' not in perf_df.columns:
        return available_model_keys[0]

    name_to_key = {
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost',
        'LSTM': 'lstm',
        'RNN': 'rnn'
    }

    ranked = perf_df.sort_values('R² Score', ascending=False)['Model'].astype(str).tolist()
    for model_name in ranked:
        mapped = name_to_key.get(model_name)
        if mapped in available_model_keys:
            return mapped
    return available_model_keys[0]

def generate_predictions(X_scaled, selected_model_keys, reg_models, deep_models):
    """Generate predictions for selected models with per-model error capture."""
    predictions = {}
    errors = []

    for model_key in selected_model_keys:
        if model_key in reg_models:
            try:
                predictions[MODEL_SHORT_NAMES[model_key]] = float(reg_models[model_key].predict(X_scaled)[0])
            except Exception as e:
                errors.append(f"{MODEL_LABELS[model_key]} failed: {e}")
        elif model_key in deep_models:
            try:
                model_obj = deep_models[model_key]
                model_input = X_scaled
                if hasattr(model_obj, 'input_shape') and model_obj.input_shape is not None and len(model_obj.input_shape) == 3:
                    model_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                pred = model_obj.predict(model_input, verbose=0)
                predictions[MODEL_SHORT_NAMES[model_key]] = float(np.ravel(pred)[0])
            except Exception as e:
                errors.append(f"{MODEL_LABELS[model_key]} failed: {e}")

    return predictions, errors

def validate_feature_ranges(features, info):
    """Return out-of-range warnings when feature metadata is available."""
    if not info or 'feature_ranges' not in info:
        return []

    warnings_list = []
    ranges = info['feature_ranges']
    for feature in FEATURE_ORDER:
        if feature not in ranges:
            continue

        min_v = ranges[feature].get('min')
        max_v = ranges[feature].get('max')
        value = float(features[feature])

        if min_v is not None and value < float(min_v):
            warnings_list.append(
                f"{feature.replace('_', ' ').title()} is below training range ({value:,.2f} < {float(min_v):,.2f})."
            )
        if max_v is not None and value > float(max_v):
            warnings_list.append(
                f"{feature.replace('_', ' ').title()} is above training range ({value:,.2f} > {float(max_v):,.2f})."
            )

    return warnings_list

def get_primary_explainability_model(models_dict):
    """Pick a stable traditional model for explainability charts."""
    if 'xgboost' in models_dict:
        return 'xgboost', models_dict['xgboost']
    if 'random_forest' in models_dict:
        return 'random_forest', models_dict['random_forest']
    return None, None

def get_explainability_model(selected_key, reg_models, deep_model_keys, deep_models_dict):
    """Resolve explainability model from current sidebar selection."""
    if selected_key in reg_models:
        return selected_key, reg_models[selected_key], None

    if selected_key in (deep_model_keys or []):
        if selected_key not in deep_models_dict:
            loaded_model, load_error = load_deep_learning_model(selected_key)
            if loaded_model is None:
                return None, None, (load_error or f"Unable to load selected model: {selected_key}")
            deep_models_dict[selected_key] = loaded_model
        return selected_key, deep_models_dict[selected_key], None

    fallback_key, fallback_model = get_primary_explainability_model(reg_models)
    if fallback_model is None:
        return None, None, "No compatible model is available for explainability."
    return fallback_key, fallback_model, None

def predict_single_model(model_key, model_obj, X_scaled):
    """Run a single prediction while handling 2D and 3D model input shapes."""
    if model_key in ('lstm', 'rnn'):
        model_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        pred = model_obj.predict(model_input, verbose=0)
        return float(np.ravel(pred)[0])
    return float(model_obj.predict(X_scaled)[0])

def get_model_feature_importance(model, feature_names):
    """Extract normalized feature importance from fitted model attributes."""
    importance_values = None

    if hasattr(model, 'feature_importances_'):
        importance_values = np.array(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        importance_values = np.abs(np.ravel(model.coef_))
    elif hasattr(model, 'layers'):
        for layer in model.layers:
            if not hasattr(layer, 'get_weights'):
                continue
            weights = layer.get_weights()
            if not weights:
                continue
            kernel = weights[0]
            if isinstance(kernel, np.ndarray) and kernel.ndim == 2 and kernel.shape[0] == len(feature_names):
                importance_values = np.mean(np.abs(kernel), axis=1)
                break

    if importance_values is None or len(importance_values) != len(feature_names):
        return None

    total = float(np.sum(importance_values))
    if total > 0:
        importance_values = importance_values / total

    return pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in feature_names],
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)

def get_emissions_mix_dataframe(features):
    """Build a ranked emissions mix table for the current scenario."""
    driver_labels = {
        'coal_co2': 'Coal Carbon',
        'oil_co2': 'Oil Carbon',
        'gas_co2': 'Gas Carbon',
        'methane': 'Methane',
        'nitrous_oxide': 'Nitrous Oxide'
    }
    rows = []
    total = sum(float(features.get(feature, 0)) for feature in driver_labels)

    for feature, label in driver_labels.items():
        value = float(features.get(feature, 0))
        rows.append({
            'Driver': label,
            'Value (MT)': value,
            'Share (%)': (value / total * 100) if total > 0 else 0.0
        })

    return pd.DataFrame(rows).sort_values('Value (MT)', ascending=False).reset_index(drop=True)

def get_scenario_intelligence(features, predicted_co2_mt, actual_co2_mt=None):
    """Compute business-style scenario metrics for the current prediction."""
    predicted_value = max(float(predicted_co2_mt), 0.0)
    population = max(float(features.get('population', 0)), 1.0)
    gdp = float(features.get('gdp', 0))
    energy_consumption = float(features.get('primary_energy_consumption', 0))
    mix_df = get_emissions_mix_dataframe(features)

    top_driver = mix_df.iloc[0]['Driver'] if not mix_df.empty else 'N/A'
    top_share = float(mix_df.iloc[0]['Share (%)']) if not mix_df.empty else 0.0

    delta_absolute = None
    delta_percent = None
    if actual_co2_mt is not None:
        delta_absolute = predicted_value - float(actual_co2_mt)
        if float(actual_co2_mt) != 0:
            delta_percent = (delta_absolute / float(actual_co2_mt)) * 100

    return {
        # Convert MT total to tonnes/person so the metric is interpretable.
        'predicted_per_capita': (predicted_value * 1_000_000) / population,
        'predicted_per_billion_gdp': (predicted_value / (gdp / 1_000_000_000)) if gdp > 0 else None,
        'predicted_per_ej': (predicted_value / energy_consumption) if energy_consumption > 0 else None,
        'top_driver': top_driver,
        'top_driver_share': top_share,
        'delta_absolute': delta_absolute,
        'delta_percent': delta_percent,
        'mix_df': mix_df
    }

def render_comparison_dashboard(predictions_map, final_prediction_mt, scenario_intel, actual_co2_mt, performance_df):
    """Render comparison-focused charts for current prediction context."""
    st.markdown("<h3 style='color: #0f3460; margin-top: 26px; font-weight: 700;'>Comparison Graphs</h3>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        mix_df = (scenario_intel or {}).get('mix_df')
        if mix_df is not None and not mix_df.empty:
            mix_plot_df = mix_df.copy()
            fig_mix = px.pie(
                mix_plot_df,
                names='Driver',
                values='Value (MT)',
                hole=0.45,
                color='Driver',
                color_discrete_sequence=['#0ea5e9', '#0284c7', '#14b8a6', '#22c55e', '#f59e0b'],
                title='Current Emissions Mix Share'
            )
            fig_mix.update_layout(template='plotly_white', height=390, legend_title='Driver')
            fig_mix.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_mix, use_container_width=True)
        else:
            st.info("Emissions mix is not available for charting.")

    with chart_col2:
        pred_vals = list((predictions_map or {}).values())
        if pred_vals:
            spread_df = pd.DataFrame({
                'Statistic': ['Min Model', 'Mean (Final)', 'Max Model'],
                'Carbon (MT)': [float(np.min(pred_vals)), float(final_prediction_mt), float(np.max(pred_vals))]
            })
            fig_spread = px.bar(
                spread_df,
                x='Statistic',
                y='Carbon (MT)',
                color='Statistic',
                color_discrete_sequence=['#14b8a6', '#0ea5e9', '#0284c7'],
                title='Prediction Spread Summary'
            )
            fig_spread.update_layout(template='plotly_white', height=390, showlegend=False)
            st.plotly_chart(fig_spread, use_container_width=True)
        else:
            st.info("Prediction spread is unavailable.")

    if mix_df is not None and not mix_df.empty:
        pareto_df = mix_df.sort_values('Value (MT)', ascending=False).reset_index(drop=True).copy()
        total_value = float(pareto_df['Value (MT)'].sum())
        if total_value > 0:
            pareto_df['Cumulative (%)'] = pareto_df['Value (MT)'].cumsum() / total_value * 100

            fig_pareto = go.Figure()
            fig_pareto.add_trace(
                go.Bar(
                    x=pareto_df['Driver'],
                    y=pareto_df['Value (MT)'],
                    name='Value (MT)',
                    marker_color='#0ea5e9'
                )
            )
            fig_pareto.add_trace(
                go.Scatter(
                    x=pareto_df['Driver'],
                    y=pareto_df['Cumulative (%)'],
                    name='Cumulative Share (%)',
                    mode='lines+markers',
                    line=dict(color='#f97316', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                )
            )
            fig_pareto.update_layout(
                title='Emissions Driver Pareto Comparison',
                template='plotly_white',
                height=390,
                xaxis_title='Driver',
                yaxis=dict(title='Value (MT)'),
                yaxis2=dict(title='Cumulative Share (%)', overlaying='y', side='right', range=[0, 100]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

def get_current_run_short_models(selected_key, scope_value, available_keys):
    """Resolve currently included short model names using last prediction when available."""
    last_predictions = st.session_state.get('last_predictions')
    if isinstance(last_predictions, dict) and last_predictions:
        return set(last_predictions.keys())

    if scope_value == 'All available models':
        return {MODEL_SHORT_NAMES.get(model_key, model_key) for model_key in (available_keys or [])}

    if selected_key:
        return {MODEL_SHORT_NAMES.get(selected_key, selected_key)}

    return set()

def render_prediction_aware_model_graphs(performance_df, current_run_models):
    """Render prediction-aware model comparison graphs in explainability section."""
    st.markdown("<h3 style='color: #0f4c81; font-weight: 700;'>Prediction-Aware Model Comparison</h3>", unsafe_allow_html=True)

    if performance_df is None or performance_df.empty or 'Model' not in performance_df.columns or 'R² Score' not in performance_df.columns:
        st.info("Model performance data is unavailable for comparison graphs.")
        return

    name_to_short = {
        'Random Forest': 'RF',
        'XGBoost': 'XGB',
        'LSTM': 'LSTM',
        'RNN': 'RNN'
    }

    perf_plot_df = performance_df.copy()
    perf_plot_df['Short Model'] = perf_plot_df['Model'].map(name_to_short)
    perf_plot_df = perf_plot_df.dropna(subset=['Short Model'])
    if perf_plot_df.empty:
        st.info("No mappable model names found for comparison charts.")
        return

    perf_plot_df['Current Run'] = perf_plot_df['Short Model'].apply(
        lambda model_name: 'Included' if model_name in (current_run_models or set()) else 'Available'
    )

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig_perf = px.bar(
            perf_plot_df,
            x='Short Model',
            y='R² Score',
            color='Current Run',
            barmode='group',
            color_discrete_map={'Included': '#0ea5e9', 'Available': '#94a3b8'},
            title='Validation R² Comparison by Model'
        )
        fig_perf.update_layout(
            template='plotly_white',
            height=360,
            xaxis_title='Model',
            yaxis_title='R² Score'
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with fig_col2:
        if 'MAE' in perf_plot_df.columns and 'RMSE' in perf_plot_df.columns:
            error_df = perf_plot_df.dropna(subset=['MAE', 'RMSE']).copy()
            if not error_df.empty:
                fig_error = px.scatter(
                    error_df,
                    x='RMSE',
                    y='MAE',
                    color='Current Run',
                    text='Short Model',
                    size='R² Score',
                    size_max=24,
                    color_discrete_map={'Included': '#0ea5e9', 'Available': '#94a3b8'},
                    title='Model Error Trade-off (MAE vs RMSE)'
                )
                fig_error.update_traces(textposition='top center')
                fig_error.update_layout(
                    template='plotly_white',
                    height=390,
                    xaxis_title='RMSE (lower is better)',
                    yaxis_title='MAE (lower is better)'
                )
                st.plotly_chart(fig_error, use_container_width=True)
            else:
                st.info("MAE/RMSE values are unavailable for scatter comparison.")
        else:
            st.info("MAE or RMSE columns are missing in performance data.")

def get_research_overview(metadata_obj, performance_df, profiles):
    """Build concise research-facing summary metrics."""
    best_model = "N/A"
    best_r2 = None
    if performance_df is not None and not performance_df.empty:
        best_row = performance_df.sort_values('R² Score', ascending=False).iloc[0]
        best_model = str(best_row['Model'])
        best_r2 = float(best_row['R² Score'])

    trained_country_count = len(metadata_obj.get('countries_trained', [])) if metadata_obj else 0
    available_country_count = len(profiles)
    feature_count = len(FEATURE_ORDER)

    return {
        'best_model': best_model,
        'best_r2': best_r2,
        'trained_country_count': trained_country_count,
        'available_country_count': available_country_count,
        'feature_count': feature_count,
        'lookback_lstm': metadata_obj.get('lookback_lstm') if metadata_obj else None,
        'train_test_split': metadata_obj.get('train_test_split') if metadata_obj else None,
        'training_date': metadata_obj.get('training_date') if metadata_obj else None
    }

def render_stat_card(label, value, meta):
    st.markdown(
        f"""
        <div class='stat-card'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
            <div class='stat-meta'>{meta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_colored_metric(label, value, tone="info", subtext=None):
    safe_label = escape(str(label))
    safe_value = escape(str(value))
    safe_subtext = escape(str(subtext)) if subtext else ""
    tone_name = tone if tone in {"success", "info", "warning", "danger", "neutral"} else "info"

    st.markdown(
        f"""
        <div class='metric-card metric-tone-{tone_name}'>
            <div class='metric-label'>{safe_label}</div>
            <div class='metric-value'>{safe_value}</div>
            {f"<div class='metric-subtext'>{safe_subtext}</div>" if safe_subtext else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_feature_range_summary(info):
    """Create short feature range summaries for research UI."""
    if not info or 'feature_ranges' not in info:
        return []

    summaries = []
    for feature in FEATURE_ORDER:
        feature_meta = info['feature_ranges'].get(feature)
        if not feature_meta:
            continue
        summaries.append(
            f"{feature.replace('_', ' ').title()}: {float(feature_meta.get('min', 0)):,.2f} to {float(feature_meta.get('max', 0)):,.2f}"
        )
    return summaries

def build_scenario_snapshot(features, predicted_co2_mt, model_key, country_name):
    """Capture a compact, comparable snapshot of the current scenario."""
    return {
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'country': country_name or 'Custom',
        'model_key': model_key,
        'predicted_co2_mt': float(predicted_co2_mt),
        'features': {feature: float(features.get(feature, 0.0)) for feature in FEATURE_ORDER}
    }

def get_scenario_delta_chart_df(snapshot_a, snapshot_b):
    """Build dataframe for Scenario B - Scenario A emissions deltas."""
    labels = {
        'coal_co2': 'Coal Carbon',
        'oil_co2': 'Oil Carbon',
        'gas_co2': 'Gas Carbon',
        'methane': 'Methane',
        'nitrous_oxide': 'Nitrous Oxide',
        'primary_energy_consumption': 'Primary Energy (EJ)',
        'predicted_co2_mt': 'Predicted Total Carbon (MT)'
    }

    rows = []
    for key, label in labels.items():
        if key == 'predicted_co2_mt':
            a_val = float(snapshot_a.get('predicted_co2_mt', 0.0))
            b_val = float(snapshot_b.get('predicted_co2_mt', 0.0))
        else:
            a_val = float(snapshot_a['features'].get(key, 0.0))
            b_val = float(snapshot_b['features'].get(key, 0.0))

        delta = b_val - a_val
        rows.append({
            'Metric': label,
            'Scenario A': a_val,
            'Scenario B': b_val,
            'Delta': delta,
            'Direction': 'Increase' if delta >= 0 else 'Decrease'
        })

    return pd.DataFrame(rows)

def get_rule_based_reduction_suggestions(features):
    """Fallback advisor when LLM is unavailable."""
    energy_features = ['coal_co2', 'oil_co2', 'gas_co2', 'methane', 'nitrous_oxide']
    ranked = sorted(energy_features, key=lambda x: float(features.get(x, 0)), reverse=True)
    target_map = {
        0: "18-25%",
        1: "12-18%",
        2: "8-12%"
    }

    suggestions = []
    for rank_idx, feat in enumerate(ranked[:3]):
        value = float(features.get(feat, 0))
        label = feat.replace('_', ' ').title()
        reduction_target = target_map.get(rank_idx, "8-12%")
        if feat == 'coal_co2':
            action = "shift power generation from coal to renewables"
        elif feat == 'oil_co2':
            action = "reduce oil-heavy transport and improve EV adoption"
        elif feat == 'gas_co2':
            action = "improve industrial efficiency and reduce gas leakage"
        elif feat == 'methane':
            action = "tighten methane leak detection and capture"
        else:
            action = "optimize fertilizer and industrial process controls"

        suggestions.append(
            f"- Prioritize {label} ({value:,.1f}): {action}; target reduction: {reduction_target} in the next 12 months."
        )

    suggestions.append("- Improve primary energy efficiency with grid modernization and efficiency standards; target 6-10% reduction.")
    suggestions.append("- Track progress quarterly and keep combined reduction target at 15-22% across the top two drivers.")
    return "\n".join(suggestions)

def run_openai_chat(api_key, model_name, system_prompt, user_prompt, temperature=0.3, max_tokens=450):
    """Run chat completion across new and legacy OpenAI SDK interfaces."""
    if OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return (response.choices[0].message.content or "").strip()

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message['content'].strip()

def classify_openai_error(err_text):
    """Return user-friendly reason for known OpenAI API failures."""
    text = (err_text or "").lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text or "error code: 429" in text:
        return "OpenAI quota exceeded (429 insufficient_quota)."
    if "invalid_api_key" in text or "incorrect api key" in text:
        return "Invalid OpenAI API key."
    if "model" in text and "not found" in text:
        return "Configured model is unavailable for this API key."
    return err_text

def get_rule_based_followup_response(user_question, features):
    """Provide a practical follow-up answer when LLM is unavailable."""
    q = (user_question or "").lower()
    emission_drivers = ['coal_co2', 'oil_co2', 'gas_co2', 'methane', 'nitrous_oxide']
    ranked = sorted(emission_drivers, key=lambda k: float(features.get(k, 0)), reverse=True)

    driver_aliases = {
        'coal_co2': ['coal', 'coal co2'],
        'oil_co2': ['oil', 'oil co2'],
        'gas_co2': ['gas', 'gas co2', 'natural gas'],
        'methane': ['methane', 'ch4'],
        'nitrous_oxide': ['nitrous', 'nitrous oxide', 'n2o']
    }

    selected_driver = None
    for driver, aliases in driver_aliases.items():
        if any(alias in q for alias in aliases):
            selected_driver = driver
            break

    if selected_driver is None:
        selected_driver = ranked[0]

    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    months_match = re.search(r"(\d+)\s*(?:month|months|mo|mos)\b", q)

    target_percent = float(pct_match.group(1)) if pct_match else None
    target_months = int(months_match.group(1)) if months_match else (12 if '12 month' in q or '1 year' in q else None)

    if target_percent is not None and target_months is not None:
        driver_value = float(features.get(selected_driver, 0))
        reduction = driver_value * (target_percent / 100.0)
        driver_label = selected_driver.replace('_', ' ').title()

        if target_percent <= 12:
            feasibility = "realistic"
        elif target_percent <= 18:
            feasibility = "possible but stretch"
        else:
            feasibility = "aggressive and requires strong policy support"

        second_driver = ranked[1] if len(ranked) > 1 else ranked[0]
        second_label = second_driver.replace('_', ' ').title()

        return (
            f"Yes, a {target_percent:.1f}% reduction in {driver_label} over {target_months} months is {feasibility} if phased properly.\n"
            f"- Target: reduce {driver_label} by about {reduction:,.1f} from the current level ({driver_value:,.1f}).\n"
            "- Months 1-3: capture quick wins through efficiency and dispatch optimization.\n"
            "- Months 4-8: shift fuel mix and scale operational controls to sustain monthly cuts.\n"
            f"- Months 9-{target_months}: lock in policy/compliance measures and monitor against monthly checkpoints.\n"
            f"- Next quarter sequencing: after stabilizing {driver_label}, start reductions in {second_label}.\n"
            "- Keep a 1-2% buffer to absorb seasonal demand variability."
        )

    top_driver = ranked[0].replace('_', ' ').title()
    return (
        "LLM follow-up is currently unavailable. "
        f"Start with your top driver ({top_driver}) and apply the recommended percentage range first, "
        "then sequence the second-largest source for the next quarter."
    )

def get_llm_reduction_suggestions(features, predicted_co2_mt):
    """Generate reduction plan with OpenAI; fallback to deterministic rules."""
    api_key = os.getenv('OPENAI_API_KEY', '').strip() or OPENAI_API_KEY_HARDCODED.strip()
    if not api_key:
        return get_rule_based_reduction_suggestions(features), "rule-based", "OPENAI_API_KEY is missing"

    try:
        model_name = os.getenv('OPENAI_MODEL', '').strip() or OPENAI_MODEL_HARDCODED
        prompt = f"""
You are a climate policy and carbon optimization advisor.

Current country profile:
- population: {features['population']}
- gdp: {features['gdp']}
- coal_co2: {features['coal_co2']}
- oil_co2: {features['oil_co2']}
- gas_co2: {features['gas_co2']}
- methane: {features['methane']}
- nitrous_oxide: {features['nitrous_oxide']}
- primary_energy_consumption: {features['primary_energy_consumption']}
- predicted_total_carbon_mt: {predicted_co2_mt:.2f}

Task:
1. Identify top 3 usage drivers to reduce first.
2. Give specific actions for each driver.
3. For each driver, include a percentage reduction target range for 12 months.
4. Estimate relative impact (high/medium/low) for each action.
4. End with a 90-day action plan in 3 bullets.

Keep response concise and practical.
Output each driver line in this format:
- <driver>: <action> | target reduction: <x-y%> | impact: <high/medium/low>
"""

        content = run_openai_chat(
            api_key=api_key,
            model_name=model_name,
            system_prompt="You provide practical carbon reduction recommendations.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=450
        )
        return content, f"openai:{model_name}", ""
    except Exception as e:
        return get_rule_based_reduction_suggestions(features), "rule-based", str(e)

def get_followup_advisor_response(user_question, features, predicted_co2_mt, current_plan):
    """Answer user follow-up questions about reduction strategy."""
    api_key = os.getenv('OPENAI_API_KEY', '').strip() or OPENAI_API_KEY_HARDCODED.strip()
    if not api_key:
        return (
            get_rule_based_followup_response(user_question, features),
            "rule-based",
            "OPENAI_API_KEY is missing"
        )

    try:
        model_name = os.getenv('OPENAI_MODEL', '').strip() or OPENAI_MODEL_HARDCODED

        prompt = f"""
You are a practical carbon reduction advisor.

Current profile:
- population: {features['population']}
- gdp: {features['gdp']}
- coal_co2: {features['coal_co2']}
- oil_co2: {features['oil_co2']}
- gas_co2: {features['gas_co2']}
- methane: {features['methane']}
- nitrous_oxide: {features['nitrous_oxide']}
- primary_energy_consumption: {features['primary_energy_consumption']}
- predicted_total_carbon_mt: {predicted_co2_mt:.2f}

Current reduction plan:
{current_plan}

User follow-up question:
{user_question}

Answer in 4-8 bullet points. Include concrete actions and percentage targets when relevant.
"""

        content = run_openai_chat(
            api_key=api_key,
            model_name=model_name,
            system_prompt="You answer climate strategy follow-up questions with practical steps.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=350
        )
        return content, f"openai:{model_name}", ""
    except Exception as e:
        return (
            get_rule_based_followup_response(user_question, features),
            "rule-based",
            classify_openai_error(str(e))
        )

# ===========================
# SIDEBAR - CONFIGURATION
# ===========================

with st.sidebar:
    st.markdown("### Configuration")
    st.divider()

    st.markdown("**Experience Mode**")
    mode = st.radio(
        "Select Mode:",
        options=['Quick Predict', 'Model Explainability'],
        help="Choose the right mode for your use case",
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Prediction Models**")
    available_model_keys = get_available_model_keys(regression_models, available_deep_model_keys)
    recommended_model_key = get_best_available_model_key(all_perf_df, available_model_keys)
    selected_model_index = available_model_keys.index(recommended_model_key) if recommended_model_key in available_model_keys else 0
    selected_model_key = st.selectbox(
        "Choose model:",
        options=available_model_keys,
        index=selected_model_index,
        format_func=lambda k: MODEL_LABELS.get(k, k),
        help="This selected model will be used for prediction. The default is the best validation R² available.",
        label_visibility="collapsed"
    )
    if recommended_model_key:
        st.caption(f"Recommended model: {MODEL_LABELS.get(recommended_model_key, recommended_model_key)}")

    prediction_scope = st.radio(
        "Prediction scope",
        options=['Selected model only', 'All available models'],
        index=0,
        help="Use one model, or run all currently available models and return a combined estimate."
    )
    st.caption("Available models: " + ", ".join(MODEL_LABELS.get(k, k) for k in available_model_keys))

    st.divider()

# ===========================
# MAIN CONTENT
# ===========================

if mode == 'Quick Predict':
    
    if 'features' not in st.session_state:
        st.session_state.features = DEFAULT_FEATURES.copy()
    for state_key, default_value in PREDICTION_STATE_DEFAULTS.items():
        if state_key not in st.session_state:
            if isinstance(default_value, list):
                st.session_state[state_key] = default_value.copy()
            else:
                st.session_state[state_key] = default_value
    if 'active_country' not in st.session_state:
        st.session_state.active_country = None
    if 'active_country_year' not in st.session_state:
        st.session_state.active_country_year = None
    if 'active_country_actual_co2' not in st.session_state:
        st.session_state.active_country_actual_co2 = None
    if 'active_country_trained' not in st.session_state:
        st.session_state.active_country_trained = False
    if 'scenario_snapshots' not in st.session_state:
        st.session_state.scenario_snapshots = []
    if 'snapshot_counter' not in st.session_state:
        st.session_state.snapshot_counter = 0
    country_options = ['Custom'] + available_countries
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = 'Custom'
    if st.session_state.selected_country not in country_options:
        st.session_state.selected_country = 'Custom'

    if st.session_state.get('auto_adjust_notice'):
        st.info(st.session_state.auto_adjust_notice)
        st.session_state.auto_adjust_notice = ""

    st.markdown("<h3 style='color: #0f4c81; font-weight: 700;'>Country Profile</h3>", unsafe_allow_html=True)
    st.selectbox(
        "Select country profile",
        options=country_options,
        key='selected_country',
        on_change=handle_country_change,
        help="Choose a trained country to auto-fill the latest available profile, or keep Custom for manual inputs."
    )

    if st.session_state.selected_country == 'Custom':
        st.error("Country selection is required.")

    if st.session_state.active_country:
        profile_parts = [f"Using latest available profile for {st.session_state.active_country}"]
        if st.session_state.active_country_year is not None:
            profile_parts.append(f"year {int(st.session_state.active_country_year)}")
        if st.session_state.active_country_actual_co2 is not None:
            profile_parts.append(f"historical carbon {st.session_state.active_country_actual_co2:,.1f} MT")
        profile_text = escape(" | ".join(profile_parts))
        st.markdown(
            f"""
            <div class='insight-box'>
                <span style='color: #0f172a; font-weight: 600;'>{profile_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.form("quick_predict_form"):
        top_row_cols = st.columns(3)
        with top_row_cols[0]:
            input_year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=int(st.session_state.features.get('year', 2022)),
                step=1
            )
        with top_row_cols[1]:
            input_population = st.number_input(
                "Population",
                min_value=0,
                value=int(st.session_state.features['population']),
                step=1000000
            )
        with top_row_cols[2]:
            input_gdp = st.number_input(
                "GDP (USD)",
                min_value=0,
                value=int(st.session_state.features['gdp']),
                step=100000000000
            )

        st.markdown("<h3 style='color: #0f4c81; font-weight: 700;'>Emissions and Energy</h3>", unsafe_allow_html=True)
        feature_cols = st.columns(2)

        with feature_cols[0]:
            input_coal_co2 = st.number_input("Coal Carbon", min_value=0.0, value=float(st.session_state.features['coal_co2']), step=10.0)
            input_oil_co2 = st.number_input("Oil Carbon", min_value=0.0, value=float(st.session_state.features['oil_co2']), step=10.0)
            input_gas_co2 = st.number_input("Gas Carbon", min_value=0.0, value=float(st.session_state.features['gas_co2']), step=10.0)

        with feature_cols[1]:
            input_methane = st.number_input("Methane", min_value=0.0, value=float(st.session_state.features['methane']), step=5.0)
            input_nitrous_oxide = st.number_input("Nitrous Oxide", min_value=0.0, value=float(st.session_state.features['nitrous_oxide']), step=5.0)
            input_primary_energy = st.number_input("Primary Energy (EJ)", min_value=0.0, value=float(st.session_state.features['primary_energy_consumption']), step=5.0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button("Predict Carbon Emissions", use_container_width=True, type="primary")

    st.divider()
    
    if predict_btn:
        try:
            if st.session_state.selected_country == 'Custom':
                st.error("Country selection is required.")
                st.stop()
            
            previous_features = st.session_state.features.copy()
            submitted_features = {
                'year': int(input_year),
                'population': float(input_population),
                'gdp': float(input_gdp),
                'coal_co2': float(input_coal_co2),
                'oil_co2': float(input_oil_co2),
                'gas_co2': float(input_gas_co2),
                'methane': float(input_methane),
                'nitrous_oxide': float(input_nitrous_oxide),
                'primary_energy_consumption': float(input_primary_energy)
            }

            adjusted_features, auto_adjusted_fields = auto_adjust_inputs(previous_features, submitted_features)
            st.session_state.features.update(adjusted_features)

            if auto_adjusted_fields:
                pretty_fields = ", ".join(field.replace('_', ' ') for field in auto_adjusted_fields)
                st.session_state.auto_adjust_notice = (
                    f"Inputs auto-adjusted from previous year/population context: {pretty_fields}. "
                    "Review updated values, then click Predict Carbon Emissions to generate output."
                )
                st.rerun()

            if scaler is None:
                st.error("Scaler is not available, so prediction cannot run.")
                st.stop()

            if not selected_model_key:
                st.error("Select a model from the sidebar to run prediction.")
                st.stop()

            if prediction_scope == 'All available models':
                selected_model_keys = list(available_model_keys)
            else:
                selected_model_keys = [selected_model_key]

            # Lazy-load selected deep learning models before inference.
            for model_key in selected_model_keys:
                if model_key in available_deep_model_keys and model_key not in dl_models:
                    loaded_dl_model, dl_model_error = load_deep_learning_model(model_key)
                    if loaded_dl_model is None:
                        st.error(dl_model_error or f"Unable to load selected deep learning model: {model_key}")
                        st.stop()
                    dl_models[model_key] = loaded_dl_model

            selected_country_name = st.session_state.active_country or st.session_state.get('selected_country') or 'Custom'
            model_features = build_model_features(st.session_state.features, selected_country_name, feature_info)

            X_input = np.array([model_features[f] for f in MODEL_FEATURE_ORDER], dtype=float)
            X_scaled = scaler.transform(X_input.reshape(1, -1))
            X_scaled = apply_feature_weighting(X_scaled, feature_info)
            
            # Get predictions
            predictions, prediction_errors = generate_predictions(
                X_scaled,
                selected_model_keys,
                regression_models,
                dl_models
            )
            
            if predictions:
                st.session_state.last_predictions = predictions
                st.session_state.last_prediction_errors = prediction_errors
                st.session_state.last_selected_model_key = selected_model_key
                st.markdown("<div class='success-box'><span style='color: #00ff66; font-weight: 700; font-size: 1.1em;'>Prediction successful.</span></div>", unsafe_allow_html=True)
                if prediction_errors:
                    st.info("Some model outputs were skipped due to runtime errors.")
                    with st.expander("Show prediction warnings"):
                        for err in prediction_errors:
                            st.write(f"- {err}")
                
                # Calculate uncertainty
                pred_values = list(predictions.values())
                uncertainty = calculate_prediction_uncertainty(pred_values, confidence_level=0.95)
                st.session_state.last_uncertainty = uncertainty
                final_prediction = float(uncertainty['mean'])
                base_year_for_projection = int(st.session_state.active_country_year) if st.session_state.active_country_year is not None else 2022
                selected_year_for_projection = int(st.session_state.features.get('year', base_year_for_projection))
                year_delta = selected_year_for_projection - base_year_for_projection
                if year_delta != 0:
                    projection_multiplier = (1 + YEAR_PROJECTION_GROWTH) ** (year_delta * YEAR_IMPACT_MULTIPLIER)
                    final_prediction = float(final_prediction * projection_multiplier)
                st.session_state.last_prediction_mean = float(final_prediction)
                single_model_mode = len(pred_values) == 1
                selected_model_label = MODEL_LABELS.get(selected_model_key, selected_model_key)
                selected_model_value = float(next(iter(predictions.values()))) if predictions else final_prediction
                if year_delta != 0:
                    selected_model_value = float(selected_model_value * projection_multiplier)
                selected_model_perf = get_selected_model_performance(selected_model_key, all_perf_df)

                # Display main carbon prediction
                prediction_context = st.session_state.active_country or 'Custom Scenario'
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #0066ff 0%, #00d4ff 100%); padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0, 102, 255, 0.3);'>
                    <div style='color: white; text-align: center;'>
                        <p style='font-size: 0.9em; margin: 0 0 10px 0; opacity: 0.9;'>Predicted Carbon Emissions for {prediction_context}</p>
                        <h2 style='color: #00ff66; font-size: 2.5em; margin: 0; font-weight: 800;'>{final_prediction:,.0f} MT</h2>
                        <p style='font-size: 0.85em; margin: 8px 0 0 0; opacity: 0.9;'>{('Model: ' + selected_model_label) if single_model_mode else 'Prediction generated'}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if not single_model_mode:
                    st.markdown("#### Model-wise carbon predictions")
                    model_rows = []
                    for short_name, pred_value in predictions.items():
                        model_rows.append({
                            'Model': short_name,
                            'Prediction (MT)': float(pred_value)
                        })
                    model_df = pd.DataFrame(model_rows).sort_values('Prediction (MT)', ascending=False)
                    
                    fig_models = px.line(
                        model_df,
                        x='Model',
                        y='Prediction (MT)',
                        markers=True,
                        title='Carbon Emissions Prediction by Model',
                        color_discrete_sequence=['#0ea5e9'],
                        labels={'Prediction (MT)': 'Predicted Carbon (MT)'}
                    )
                    fig_models.update_traces(
                        marker=dict(size=12),
                        line=dict(width=3)
                    )
                    fig_models.update_layout(
                        height=400,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_models, use_container_width=True)

                    # Fallback block if multiple predictions are ever passed in future.
                    st.markdown("<h4 style='color: #00d4ff; font-weight: 700; margin-top: 20px;'>Selected Model Carbon Prediction:</h4>", unsafe_allow_html=True)
                    render_colored_metric(selected_model_label, f"{selected_model_value:,.0f} MT", tone="info")
                
                # Insights
                st.markdown("<h3 style='color: #ff9900; font-weight: 700; margin-top: 30px;'>Prediction Insights</h3>", unsafe_allow_html=True)
                col_i1, col_i2 = st.columns(2)
                
                if single_model_mode:
                    with col_i1:
                        render_colored_metric("Model Used", selected_model_label, tone="info")
                    with col_i2:
                        if selected_model_perf and selected_model_perf.get('r2') is not None:
                            r2_value = selected_model_perf['r2']
                            if r2_value >= 0.8:
                                reliability = "High"
                                reliability_tone = "success"
                            elif r2_value >= 0.5:
                                reliability = "Medium"
                                reliability_tone = "warning"
                            elif r2_value >= 0:
                                reliability = "Low"
                                reliability_tone = "warning"
                            else:
                                reliability = "Poor"
                                reliability_tone = "danger"
                            render_colored_metric("Validation Reliability", reliability, tone=reliability_tone, subtext=f"R² = {r2_value:.3f}")
                        else:
                            render_colored_metric("Validation Reliability", "Unknown", tone="neutral")
                else:
                    with col_i1:
                        agreement = 100 - uncertainty['cv']
                        agreement_tone = "success" if agreement >= 80 else "warning" if agreement >= 50 else "danger"
                        render_colored_metric("Model Agreement", f"{agreement:.1f}%", tone=agreement_tone)

                    with col_i2:
                        if uncertainty['cv'] < 10:
                            risk = "Low Risk"
                            risk_tone = "success"
                        elif uncertainty['cv'] < 20:
                            risk = "Medium Risk"
                            risk_tone = "warning"
                        else:
                            risk = "High Risk"
                            risk_tone = "danger"
                        render_colored_metric("Confidence", risk, tone=risk_tone)

                scenario_intel = get_scenario_intelligence(
                    st.session_state.features,
                    final_prediction,
                    st.session_state.active_country_actual_co2
                )

                st.markdown("<h3 style='color: #0f3460; margin-top: 30px; font-weight: 700;'>Scenario Intelligence</h3>", unsafe_allow_html=True)
                intel_cols = st.columns(4)

                with intel_cols[0]:
                    if scenario_intel['delta_absolute'] is not None:
                        delta_val = scenario_intel['delta_absolute']
                        delta_pct = scenario_intel['delta_percent']
                        delta_tone = "danger" if delta_val > 0 else "success" if delta_val < 0 else "neutral"
                        render_colored_metric(
                            "Vs Latest Actual Carbon",
                            f"{delta_val:,.0f} MT",
                            tone=delta_tone,
                            subtext=f"{delta_pct:.1f}% vs actual" if delta_pct is not None else None
                        )
                    else:
                        render_colored_metric("Vs Latest Actual Carbon", "N/A", tone="neutral")

                with intel_cols[1]:
                    render_colored_metric("Carbon / Capita (t/person)", f"{scenario_intel['predicted_per_capita']:.2f}", tone="info")

                with intel_cols[2]:
                    if scenario_intel['predicted_per_billion_gdp'] is not None:
                        render_colored_metric("Carbon / $1B GDP", f"{scenario_intel['predicted_per_billion_gdp']:.2f} MT", tone="warning")
                    else:
                        render_colored_metric("Carbon / $1B GDP", "N/A", tone="neutral")

                with intel_cols[3]:
                    render_colored_metric(
                        "Primary Driver",
                        scenario_intel['top_driver'],
                        tone="danger",
                        subtext=f"{scenario_intel['top_driver_share']:.1f}% share"
                    )

                top_actions = scenario_intel['mix_df'].head(3)
                if not top_actions.empty:
                    st.markdown("""
                    <div class='insight-box'>
                    <b style='color: #0f3460;'>Priority Action Queue</b>
                    </div>
                    """, unsafe_allow_html=True)
                    for idx, row in top_actions.iterrows():
                        st.write(f"{idx + 1}. {row['Driver']} drives {row['Share (%)']:.1f}% of the current emissions mix and should be targeted first.")

                render_comparison_dashboard(
                    predictions,
                    final_prediction,
                    scenario_intel,
                    st.session_state.active_country_actual_co2,
                    all_perf_df
                )
            else:
                st.warning("No prediction was generated with the selected model.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if (not predict_btn) and st.session_state.last_predictions and st.session_state.last_uncertainty:
        st.markdown("<h3 style='color: #0066ff; font-weight: 700;'>Last Prediction Snapshot</h3>", unsafe_allow_html=True)
        st.caption("Showing the most recent prediction results. Click Predict Carbon Emissions to refresh.")

        last_predictions = st.session_state.last_predictions
        last_uncertainty = st.session_state.last_uncertainty
        last_selected_model_key = st.session_state.get('last_selected_model_key')
        single_model_snapshot = len(last_predictions) == 1

        if st.session_state.last_prediction_errors:
            with st.expander("Show prediction warnings"):
                for err in st.session_state.last_prediction_errors:
                    st.write(f"- {err}")

        snapshot_model_value = float(next(iter(last_predictions.values())))
        snapshot_model_label = MODEL_LABELS.get(last_selected_model_key, last_selected_model_key or next(iter(last_predictions.keys())))
        render_colored_metric(snapshot_model_label, f"{snapshot_model_value:,.0f} MT", tone="info")

        if single_model_snapshot:
            snap_cols = st.columns(3)
            with snap_cols[0]:
                render_colored_metric("Final", f"{st.session_state.last_prediction_mean:,.0f} MT", tone="info")
            with snap_cols[1]:
                snapshot_model_perf = get_selected_model_performance(last_selected_model_key, all_perf_df)
                if snapshot_model_perf and snapshot_model_perf.get('r2') is not None:
                    r2_value = snapshot_model_perf['r2']
                    r2_tone = "success" if r2_value >= 0.8 else "warning" if r2_value >= 0.5 else "danger"
                    render_colored_metric("Model R²", f"{r2_value:.3f}", tone=r2_tone)
                else:
                    render_colored_metric("Model R²", "N/A", tone="neutral")
            with snap_cols[2]:
                if snapshot_model_perf and snapshot_model_perf.get('rmse') is not None:
                    render_colored_metric("Model RMSE", f"{snapshot_model_perf['rmse']:.4f}", tone="info")
                else:
                    render_colored_metric("Model RMSE", "N/A", tone="neutral")
        else:
            metric_cols = st.columns(5)
            snap_metrics = [
                ("Final", f"{st.session_state.last_prediction_mean:,.0f} MT"),
                ("Std Dev", f"{last_uncertainty['std']:,.0f}"),
                ("CV", f"{last_uncertainty['cv']:.1f}%"),
                (f"{int(last_uncertainty.get('confidence_level', 0.95) * 100)}% Lower Carbon", f"{last_uncertainty.get('ci_lower', last_uncertainty['ci_95_lower']):,.0f} MT"),
                (f"{int(last_uncertainty.get('confidence_level', 0.95) * 100)}% Upper Carbon", f"{last_uncertainty.get('ci_upper', last_uncertainty['ci_95_upper']):,.0f} MT")
            ]
            for col, (label, value) in zip(metric_cols, snap_metrics):
                with col:
                    st.metric(label, value)

        # Keep insights visible across reruns (e.g., after advisor button clicks).
        st.markdown("<h3 style='color: #ff9900; font-weight: 700; margin-top: 30px;'>Prediction Insights</h3>", unsafe_allow_html=True)
        snap_i1, snap_i2 = st.columns(2)

        if single_model_snapshot:
            with snap_i1:
                render_colored_metric("Model Used", snapshot_model_label, tone="info")
            with snap_i2:
                snapshot_model_perf = get_selected_model_performance(last_selected_model_key, all_perf_df)
                if snapshot_model_perf and snapshot_model_perf.get('r2') is not None:
                    r2_value = snapshot_model_perf['r2']
                    if r2_value >= 0.8:
                        reliability = "High"
                        reliability_tone = "success"
                    elif r2_value >= 0.5:
                        reliability = "Medium"
                        reliability_tone = "warning"
                    elif r2_value >= 0:
                        reliability = "Low"
                        reliability_tone = "warning"
                    else:
                        reliability = "Poor"
                        reliability_tone = "danger"
                    render_colored_metric("Validation Reliability", reliability, tone=reliability_tone, subtext=f"R² = {r2_value:.3f}")
                else:
                    render_colored_metric("Validation Reliability", "Unknown", tone="neutral")
        else:
            with snap_i1:
                agreement = 100 - last_uncertainty['cv']
                agreement_tone = "success" if agreement >= 80 else "warning" if agreement >= 50 else "danger"
                render_colored_metric("Model Agreement", f"{agreement:.1f}%", tone=agreement_tone)

            with snap_i2:
                if last_uncertainty['cv'] < 10:
                    risk = "Low Risk"
                    risk_tone = "success"
                elif last_uncertainty['cv'] < 20:
                    risk = "Medium Risk"
                    risk_tone = "warning"
                else:
                    risk = "High Risk"
                    risk_tone = "danger"
                render_colored_metric("Confidence", risk, tone=risk_tone)

        snapshot_scenario_intel = get_scenario_intelligence(
            st.session_state.features,
            st.session_state.last_prediction_mean,
            st.session_state.active_country_actual_co2
        )

        st.markdown("<h3 style='color: #0f3460; margin-top: 30px; font-weight: 700;'>Scenario Intelligence</h3>", unsafe_allow_html=True)
        snap_intel_cols = st.columns(4)

        with snap_intel_cols[0]:
            if snapshot_scenario_intel['delta_absolute'] is not None:
                delta_val = snapshot_scenario_intel['delta_absolute']
                delta_pct = snapshot_scenario_intel['delta_percent']
                delta_tone = "danger" if delta_val > 0 else "success" if delta_val < 0 else "neutral"
                render_colored_metric(
                    "Vs Latest Actual Carbon",
                    f"{delta_val:,.0f} MT",
                    tone=delta_tone,
                    subtext=f"{delta_pct:.1f}% vs actual" if delta_pct is not None else None
                )
            else:
                render_colored_metric("Vs Latest Actual Carbon", "N/A", tone="neutral")

        with snap_intel_cols[1]:
            render_colored_metric("Carbon / Capita (t/person)", f"{snapshot_scenario_intel['predicted_per_capita']:.2f}", tone="info")

        with snap_intel_cols[2]:
            if snapshot_scenario_intel['predicted_per_billion_gdp'] is not None:
                render_colored_metric("Carbon / $1B GDP", f"{snapshot_scenario_intel['predicted_per_billion_gdp']:.2f} MT", tone="warning")
            else:
                render_colored_metric("Carbon / $1B GDP", "N/A", tone="neutral")

        with snap_intel_cols[3]:
            render_colored_metric(
                "Primary Driver",
                snapshot_scenario_intel['top_driver'],
                tone="danger",
                subtext=f"{snapshot_scenario_intel['top_driver_share']:.1f}% share"
            )

        snapshot_top_actions = snapshot_scenario_intel['mix_df'].head(3)
        if not snapshot_top_actions.empty:
            st.markdown("""
            <div class='insight-box'>
            <b style='color: #0f3460;'>Priority Action Queue</b>
            </div>
            """, unsafe_allow_html=True)
            for idx, row in snapshot_top_actions.iterrows():
                st.write(f"{idx + 1}. {row['Driver']} drives {row['Share (%)']:.1f}% of the current emissions mix and should be targeted first.")

        render_comparison_dashboard(
            last_predictions,
            st.session_state.last_prediction_mean,
            snapshot_scenario_intel,
            st.session_state.active_country_actual_co2,
            all_perf_df
        )

    if st.session_state.last_prediction_mean is not None:
        save_col, clear_col = st.columns([3, 1])
        with save_col:
            if st.button("Save Snapshot", type="secondary", use_container_width=True, key="save_snapshot_btn"):
                snapshot = build_scenario_snapshot(
                    st.session_state.features,
                    st.session_state.last_prediction_mean,
                    st.session_state.get('last_selected_model_key'),
                    st.session_state.active_country
                )
                st.session_state.snapshot_counter += 1
                snapshot['name'] = f"Scenario {st.session_state.snapshot_counter}: {snapshot['country']} ({snapshot['saved_at']})"
                st.session_state.scenario_snapshots.append(snapshot)
                if len(st.session_state.scenario_snapshots) > 10:
                    st.session_state.scenario_snapshots = st.session_state.scenario_snapshots[-10:]
                st.success("Snapshot saved for scenario comparison.")
        with clear_col:
            if st.button("Clear Saved", use_container_width=True, key="clear_saved_snapshots"):
                st.session_state.scenario_snapshots = []
                st.session_state.snapshot_counter = 0
                st.info("Saved snapshots cleared.")

        if len(st.session_state.scenario_snapshots) >= 2:
            st.markdown("<h3 style='color: #0f3460; font-weight: 700; margin-top: 20px;'>Scenario Comparison</h3>", unsafe_allow_html=True)

            snapshot_names = [item['name'] for item in st.session_state.scenario_snapshots]
            left_selector_col, right_selector_col = st.columns(2)

            with left_selector_col:
                scenario_a_name = st.selectbox(
                    "Scenario A",
                    options=snapshot_names,
                    index=max(len(snapshot_names) - 2, 0),
                    key="scenario_a_select"
                )

            with right_selector_col:
                default_b_index = len(snapshot_names) - 1
                scenario_b_name = st.selectbox(
                    "Scenario B",
                    options=snapshot_names,
                    index=default_b_index,
                    key="scenario_b_select"
                )

            snapshot_by_name = {item['name']: item for item in st.session_state.scenario_snapshots}
            scenario_a = snapshot_by_name.get(scenario_a_name)
            scenario_b = snapshot_by_name.get(scenario_b_name)

            if scenario_a_name == scenario_b_name:
                st.warning("Select two different snapshots to compare scenarios.")
            elif scenario_a and scenario_b:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"### {scenario_a['name']}")
                    render_colored_metric("Predicted Carbon", f"{scenario_a['predicted_co2_mt']:,.0f} MT", tone="info")
                    render_colored_metric("Country", scenario_a['country'], tone="neutral")
                    render_colored_metric("Model", MODEL_LABELS.get(scenario_a.get('model_key'), scenario_a.get('model_key') or 'N/A'), tone="neutral")

                with col_b:
                    st.markdown(f"### {scenario_b['name']}")
                    render_colored_metric("Predicted Carbon", f"{scenario_b['predicted_co2_mt']:,.0f} MT", tone="info")
                    render_colored_metric("Country", scenario_b['country'], tone="neutral")
                    render_colored_metric("Model", MODEL_LABELS.get(scenario_b.get('model_key'), scenario_b.get('model_key') or 'N/A'), tone="neutral")

                delta_df = get_scenario_delta_chart_df(scenario_a, scenario_b)
                total_delta = float(delta_df[delta_df['Metric'] == 'Predicted Total Carbon (MT)']['Delta'].iloc[0])
                total_delta_tone = "danger" if total_delta > 0 else "success" if total_delta < 0 else "neutral"
                render_colored_metric("Total Emissions Delta (B - A)", f"{total_delta:,.0f} MT", tone=total_delta_tone)

                fig_delta = px.bar(
                    delta_df,
                    x='Metric',
                    y='Delta',
                    color='Direction',
                    color_discrete_map={'Increase': '#dc2626', 'Decrease': '#16a34a'},
                    title='Scenario B - Scenario A Delta by Metric'
                )
                fig_delta.update_layout(template='plotly_white', height=430, xaxis_title='Metric', yaxis_title='Delta')
                fig_delta.add_hline(y=0, line_width=1, line_dash='dash', line_color='#475569')
                st.plotly_chart(fig_delta, use_container_width=True)

        st.divider()
        st.markdown("<h3 style='color: #0f3460; font-weight: 700;'>AI Carbon Reduction Advisor</h3>", unsafe_allow_html=True)
        st.caption("Get targeted suggestions on which usage areas to decrease to reduce carbon output.")

        if st.button("Generate Reduction Suggestions", type="primary", use_container_width=True, key="ai_advisor_btn"):
            with st.spinner("Analyzing current profile and generating recommendations..."):
                advice_text, advisor_source, advisor_error = get_llm_reduction_suggestions(
                    st.session_state.features,
                    st.session_state.last_prediction_mean
                )
            st.session_state.advisor_text = advice_text
            st.session_state.advisor_source = advisor_source
            st.session_state.advisor_error = advisor_error

        if st.session_state.advisor_text:
            if st.session_state.advisor_source.startswith("openai:"):
                st.success(f"Suggestions generated using {st.session_state.advisor_source}.")
            else:
                if st.session_state.advisor_error:
                    with st.expander("Advisor fallback reason"):
                        st.write(st.session_state.advisor_error)

            st.markdown("""
            <div class='insight-box'>
            <b style='color: #0f3460;'>Recommended Reduction Priorities</b>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.advisor_text)

            st.markdown("### Ask Follow-up Question")
            with st.form("advisor_followup_form", clear_on_submit=True):
                user_followup = st.text_input(
                    "Ask about implementation, timeline, sector-specific actions, or budget trade-offs:",
                    key="advisor_followup_input"
                )
                ask_advisor_btn = st.form_submit_button("Ask Advisor", use_container_width=True)

            if ask_advisor_btn:
                if user_followup.strip():
                    with st.spinner("Generating follow-up answer..."):
                        followup_answer, followup_source, followup_error = get_followup_advisor_response(
                            user_followup.strip(),
                            st.session_state.features,
                            st.session_state.last_prediction_mean,
                            st.session_state.advisor_text
                        )
                    st.session_state.advisor_chat_history.append({
                        'question': user_followup.strip(),
                        'answer': followup_answer,
                        'source': followup_source,
                        'error': followup_error
                    })
                else:
                    st.warning("Please enter a follow-up question.")

            if st.session_state.advisor_chat_history:
                st.markdown("### Advisor Conversation")
                for idx, item in enumerate(st.session_state.advisor_chat_history, start=1):
                    st.markdown(f"**You {idx}:** {item['question']}")
                    st.markdown(f"**Advisor {idx}:** {item['answer']}")
                    if item['source'].startswith("openai:"):
                        st.caption(f"Source: {item['source']}")
                    elif item['error']:
                        with st.expander(f"Follow-up fallback reason {idx}"):
                            st.write(item['error'])

elif mode == 'Model Explainability':
    st.markdown("<h2 class='section-header'>Model Explainability</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Sensitivity Analysis", "Model Comparison"])
    
    with tab1:
        st.markdown("<h3 style='color: #00d4ff; font-weight: 700;'>Feature Importance</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <span style='color: #00d4ff; font-weight: 600;'>Shows which input features have the biggest impact on carbon predictions.</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Calculate Importance", type="primary", use_container_width=True, key="feature_importance"):
            with st.spinner("Analyzing..."):
                model_name, explain_model, explain_error = get_explainability_model(
                    selected_model_key,
                    regression_models,
                    available_deep_model_keys,
                    dl_models
                )
                if explain_model is None:
                    st.error(explain_error or "No compatible model is available for feature importance.")
                else:
                    importance_df = get_model_feature_importance(explain_model, MODEL_FEATURE_ORDER)
                    if importance_df is None:
                        st.error(f"{MODEL_LABELS.get(model_name, model_name)} does not expose usable feature importance attributes.")
                    else:
                        fig_imp = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis',
                            text='Importance'
                        )
                        fig_imp.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        fig_imp.update_layout(
                            height=400,
                            template='plotly_white',
                            showlegend=False,
                            title=f"Feature Importance ({model_name})"
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab2:
        st.markdown("<h3 style='color: #ff9900; font-weight: 700;'>Sensitivity Analysis</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
        <span style='color: #ff9900; font-weight: 600;'>Analyze how carbon changes when you vary each feature.</span>
        </div>
        """, unsafe_allow_html=True)
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            feature_to_analyze = st.selectbox(
                "Select feature:",
                options=FEATURE_ORDER,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col_s2:
            variation_range = st.slider("Variation Range (%)", 10, 100, 50, step=10)
        
        if st.button("Run Analysis", type="primary", use_container_width=True, key="sensitivity_run"):
            model_name, explain_model, explain_error = get_explainability_model(
                selected_model_key,
                regression_models,
                available_deep_model_keys,
                dl_models
            )
            if explain_model is None or scaler is None:
                st.error(explain_error or "Sensitivity analysis requires a loaded model and scaler.")
            else:
                base_features = st.session_state.get('features', DEFAULT_FEATURES.copy())
                selected_country_name = st.session_state.active_country or st.session_state.get('selected_country') or 'Custom'
                base_model_features = build_model_features(base_features, selected_country_name, feature_info)

                base_input = np.array([base_model_features[f] for f in MODEL_FEATURE_ORDER], dtype=float)
                feature_idx = MODEL_FEATURE_ORDER.index(feature_to_analyze)
                base_value = base_input[feature_idx]

                lower = base_value * (1 - variation_range / 100)
                upper = base_value * (1 + variation_range / 100)
                variations = np.linspace(lower, upper, 20)
                predictions = []

                for var in variations:
                    modified_input = base_input.copy()
                    modified_input[feature_idx] = var
                    modified_scaled = scaler.transform(modified_input.reshape(1, -1))
                    modified_scaled = apply_feature_weighting(modified_scaled, feature_info)
                    predictions.append(predict_single_model(model_name, explain_model, modified_scaled))

                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(
                    x=variations,
                    y=predictions,
                    mode='lines+markers',
                    name='Prediction',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8)
                ))
                fig_sens.add_vline(
                    x=base_value,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text="Current"
                )

                fig_sens.update_layout(
                    title=f"Sensitivity to {feature_to_analyze.replace('_', ' ').title()} ({model_name})",
                    xaxis_title="Feature Value",
                    yaxis_title="Predicted Carbon (MT)",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_sens, use_container_width=True)

    with tab3:
        current_run_models = get_current_run_short_models(
            selected_model_key,
            prediction_scope,
            available_model_keys
        )

        if st.session_state.get('last_predictions'):
            st.caption("Using latest Predict run from Quick Predict mode.")
        else:
            st.caption("No previous prediction found. Using current sidebar model selection.")

        render_prediction_aware_model_graphs(all_perf_df, current_run_models)

# Footer
st.divider()
st.markdown("""
<div class='footer'>
<p><b>Comparative Analysis of Deep Learning and Machine Learning Models</b><br>
<small>CO2 Forecasting Research</small></p>
</div>
""", unsafe_allow_html=True)
