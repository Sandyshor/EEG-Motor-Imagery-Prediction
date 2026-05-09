
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import get_file_info, load_edf_from_upload
from src.features import compute_band_power_features
from src.inference import (
    list_available_models,
    load_metadata,
    load_model,
    predict_epochs,
)
from src.preprocessing import apply_bandpass, extract_events, make_epochs
from sklearn.preprocessing import StandardScaler as _SubjScaler
from src.utils import BANDS, CLASS_LABELS, FEATURE_CHANNELS, FMAX, FMIN, MOTOR_CHANNELS, TMAX, TMIN, class_name
from src.visualisation import (
    plot_band_power_by_class,
    plot_confusion_matrix,
    plot_prediction_timeline,
    plot_psd,
    plot_raw_signal,
    plot_raw_vs_filtered,
)

# ============================================================================
# Page setup
# ============================================================================
st.set_page_config(
    page_title="EEG Motor Imagery Prediction",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 EEG Motor Imagery Prediction Application")
st.caption(
    "PhysioNet EEG Motor Movement / Imagery Dataset · "
    "Left-hand vs Right-hand motor imagery (runs 4, 8, 12)"
)


with st.sidebar:
    st.header("Upload & Controls")
    uploaded_file = st.file_uploader(
        "Upload EDF file",
        type=["edf"],
        help="Use a file from PhysioNet EEG MMI runs 4, 8, or 12.",
    )

    available_models = list_available_models()
    if not available_models:
        st.error(
            "No trained models found in `models/`. "
            "Run `python train_models.py` locally first."
        )
        model_name = None
    else:
        model_name = st.selectbox(
            "Model",
            available_models,
            help="Pre-trained model used for prediction.",
        )

    st.divider()
    st.markdown("**Ground-truth labels**")
    st.markdown("- T1 = imagined **left** fist")
    st.markdown("- T2 = imagined **right** fist")
    st.markdown("- T0 = rest (not classified)")


# ============================================================================
# No file uploaded yet — show instructions and offline summary
# ============================================================================
def render_offline_summary(metadata: dict, key_prefix: str = ""):
    """Render the offline training summary and confusion matrices."""
    if not metadata:
        st.info("Train models locally to populate this section.")
        return

    results = metadata.get("results", {})

    st.markdown(
        f"**Training subjects:** {len(metadata.get('training_subjects', []))} "
        f"(S{min(metadata.get('training_subjects', [1])):03d}–"
        f"S{max(metadata.get('training_subjects', [1])):03d}) · "
        f"**Runs:** {metadata.get('runs', [])} · "
        f"**Strategy:** {metadata.get('evaluation_strategy', '')}"
    )

    pre = metadata.get("preprocessing", {})
    feat = metadata.get("feature_extraction", {})
    st.markdown(
        f"**Preprocessing:** band-pass {pre.get('bandpass_low_hz')}–"
        f"{pre.get('bandpass_high_hz')} Hz · "
        f"epochs {pre.get('epoch_tmin_s')} → {pre.get('epoch_tmax_s')} s · "
        f"channels: {pre.get('channels')}  \n"
        f"**Features:** {feat.get('method')} · "
        f"bands {list(feat.get('bands_hz', {}).keys())} · "
        f"{feat.get('n_features_per_epoch')} features/epoch"
    )

    cols = st.columns(len(results)) if results else []
    for col, (name, res) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {name.replace('_', ' ').title()}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{res['accuracy']:.3f}")
            m2.metric("F1-score", f"{res['f1_score']:.3f}")
            st.caption(
                f"Within-subject eval · "
                f"cross-subject CV: {res['cv_accuracy_mean']:.3f} ± "
                f"{res['cv_accuracy_std']:.3f}"
            )
            cm = np.array(res["confusion_matrix"])
            fig = plot_confusion_matrix(
                cm,
                class_names=[CLASS_LABELS[1], CLASS_LABELS[2]],
                title=f"{name} – validation",
            )
            st.pyplot(fig, use_container_width=True)


metadata = load_metadata()

if uploaded_file is None:
    st.info(
        "👈 Upload an EDF file from runs 4, 8, or 12 of the "
        "[PhysioNet EEG Motor Movement/Imagery Dataset]"
        "(https://www.physionet.org/content/eegmmidb/1.0.0/) "
        "to begin analysis."
    )
    st.divider()
    st.subheader("Offline Model Summary")
    render_offline_summary(metadata)
    st.stop()


# ============================================================================
# Load the uploaded file
# ============================================================================
@st.cache_data(show_spinner="Reading EDF file...")
def cached_file_info(file_bytes: bytes, file_name: str):
    """Cache file loading on raw bytes so we don't re-read on every rerun."""
    class _Wrap:
        def __init__(self, data, name):
            self._data = data
            self.name = name
        def getvalue(self):
            return self._data
    raw = load_edf_from_upload(_Wrap(file_bytes, file_name))
    return raw, get_file_info(raw)


try:
    raw, info = cached_file_info(uploaded_file.getvalue(), uploaded_file.name)
except Exception as e:
    st.error(f"Could not read EDF file: {e}")
    st.stop()


# ============================================================================
# 1. File information
# ============================================================================
st.subheader("📄 File Information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sampling Frequency", f"{info['sampling_frequency']:.0f} Hz")
c2.metric("Channels", info["n_channels"])
c3.metric("Duration", f"{info['duration_seconds']:.1f} s")
c4.metric(
    "Annotations",
    ", ".join(info["annotation_descriptions"]) or "—",
)

with st.expander("All channel names"):
    st.code(", ".join(info["channel_names"]))


# ============================================================================
# 2. EEG Signal Visualisation
# ============================================================================
st.subheader("📈 EEG Signal Visualisation")

# Default to a motor cortex channel if present, otherwise the first channel
default_channels = [c for c in MOTOR_CHANNELS if c in info["channel_names"]]
default_channel = default_channels[0] if default_channels else info["channel_names"][0]

vc1, vc2 = st.columns([1, 3])
with vc1:
    selected_channel = st.selectbox(
        "Channel",
        info["channel_names"],
        index=info["channel_names"].index(default_channel),
    )
    duration = info["duration_seconds"]
    time_range = st.slider(
        "Time range (s)",
        min_value=0.0,
        max_value=float(duration),
        value=(0.0, min(30.0, float(duration))),
        step=1.0,
    )

with vc2:
    sfreq = info["sampling_frequency"]
    s_idx = int(time_range[0] * sfreq)
    e_idx = int(time_range[1] * sfreq)
    times = raw.times[s_idx:e_idx]

    raw_signal = raw.get_data(picks=[selected_channel])[0, s_idx:e_idx]

    # Pre-compute band-pass filtered version for the comparison plot
    raw_filt = apply_bandpass(raw)
    filt_signal = raw_filt.get_data(picks=[selected_channel])[0, s_idx:e_idx]

    # Get events for overlay
    try:
        events, _ = extract_events(raw)
    except Exception:
        events = np.array([]).reshape(0, 3)

    fig_raw = plot_raw_signal(
        times, raw_signal, selected_channel,
        events=events if len(events) else None,
        sfreq=sfreq,
    )
    st.pyplot(fig_raw, use_container_width=True)

vc3, vc4 = st.columns(2)
with vc3:
    fig_rf = plot_raw_vs_filtered(
        times, raw_signal, filt_signal, selected_channel, FMIN, FMAX
    )
    st.pyplot(fig_rf, use_container_width=True)
with vc4:
    fig_psd = plot_psd(raw_signal, sfreq, selected_channel)
    st.pyplot(fig_psd, use_container_width=True)


# ============================================================================
# 3. Preprocessing, Feature Extraction, and Model Inference
# ============================================================================
st.subheader("⚙️ Preprocessing, Feature Extraction, and Model Inference")

steps = [
    f"**1. Band-pass filter**\n\n{FMIN:g}–{FMAX:g} Hz (FIR)",
    f"**2. Extract events**\n\nT1 (left) and T2 (right)\nfrom EDF annotations",
    f"**3. Create epochs**\n\nWindow {TMIN:g} → {TMAX:g} s\naround each event",
    f"**4. Band-power features**\n\nWelch PSD → mean power\nper channel × band",
    f"**5. Pre-trained model**\n\n{model_name or '—'}\n(loaded from disk)",
]
sc = st.columns(len(steps))
for col, text in zip(sc, steps):
    col.info(text)


# Run the full pipeline on the uploaded file
@st.cache_data(show_spinner="Extracting epochs and features...")
def run_pipeline(file_bytes: bytes, file_name: str):
    class _Wrap:
        def __init__(self, data, name):
            self._data = data
            self.name = name
        def getvalue(self):
            return self._data

    raw_ = load_edf_from_upload(_Wrap(file_bytes, file_name))
    raw_filt_ = apply_bandpass(raw_)
    events_, event_id_ = extract_events(raw_filt_)
    if len(events_) == 0:
        return None
    epochs_ = make_epochs(raw_filt_, events_, event_id_)

    # (a) Full 64-channel data — used by CSP/Riemannian models (need spatial coverage)
    data_full_ = epochs_.get_data()                      # (n_epochs, 64, n_times)

    # (b) 9 motor-cortex channels — used for band-power features
    present = [c for c in FEATURE_CHANNELS if c in epochs_.ch_names]
    epochs_mc_ = epochs_.copy().pick(present) if present else epochs_
    data_mc_ = epochs_mc_.get_data()                     # (n_epochs, 9, n_times)
    n_bp_ch_ = data_mc_.shape[1]

    # Band-power features from the 9-channel data
    feats_ = compute_band_power_features(
        data_mc_, sfreq=epochs_mc_.info["sfreq"],
        channel_names=list(epochs_mc_.ch_names),
    )
    if feats_.shape[0] > 1:
        feats_ = _SubjScaler().fit_transform(feats_)

    # CSP/Riemannian raw input: the FULL 64-channel data, amplitude-normalised
    scale_ = data_full_.std() + 1e-10
    data_csp_ = (data_full_ - data_full_.mean(axis=-1, keepdims=True)) / scale_

    labels_ = epochs_.events[:, -1].astype(int)
    starts_ = epochs_.events[:, 0] / epochs_.info["sfreq"] + TMIN
    ends_ = starts_ + (TMAX - TMIN)
    return {
        "features": feats_,
        "n_bp_channels": n_bp_ch_,
        "raw_epochs": data_csp_,   # (n_epochs, 64, n_times) normalised — for CSP/Riemann
        "labels": labels_,
        "starts": starts_,
        "ends": ends_,
        "n_channels": n_bp_ch_,
    }


pipeline_out = run_pipeline(uploaded_file.getvalue(), uploaded_file.name)

if pipeline_out is None:
    st.warning(
        "This EDF file has no T1/T2 events. "
        "Please upload a file from runs 4, 8, or 12 (imagined left/right fist)."
    )
    st.divider()
    st.subheader("Offline Model Summary")
    render_offline_summary(metadata)
    st.stop()

features = pipeline_out["features"]
y_true = pipeline_out["labels"]
n_bp_ch = pipeline_out["n_bp_channels"]
n_pure_bp = n_bp_ch * len(BANDS)   # features without laterality extras

st.markdown(
    f"Extracted **{len(y_true)} epochs** "
    f"({(y_true == 1).sum()} left · {(y_true == 2).sum()} right) · "
    f"band-power feature vector = **{features.shape[1]}** "
    f"({n_bp_ch} motor-cortex channels × {len(BANDS)} bands"
    + (f" + {features.shape[1] - n_pure_bp} laterality" if features.shape[1] > n_pure_bp else "")
    + ")"
)

# Pass only the pure band-power block so the visualisation reshapes correctly
bp_fig = plot_band_power_by_class(features[:, :n_pure_bp], y_true, n_bands=len(BANDS))
st.pyplot(bp_fig, use_container_width=True)


# ============================================================================
# 4. Offline training summary
# ============================================================================
st.divider()
st.subheader("🎓 Offline Model Training Summary")
render_offline_summary(metadata, key_prefix="main_")


# ============================================================================
# 5. Predictions on the uploaded file
# ============================================================================
st.divider()
st.subheader("🔮 Prediction Results and Evaluation")

if model_name is None:
    st.error("No model available — please train models first.")
    st.stop()

try:
    model = load_model(model_name)
except Exception as e:
    st.error(f"Could not load model `{model_name}`: {e}")
    st.stop()

# CSP models expect raw 3-D epoch data; all others expect band-power features.
model_input_types = metadata.get("model_input_types", {})
needs_raw = model_input_types.get(model_name, "band_power") == "raw_epochs"

if needs_raw:
    input_data = pipeline_out["raw_epochs"]
else:
    input_data = features
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and features.shape[1] != expected:
        st.error(
            f"Feature size mismatch: this file produced {features.shape[1]} "
            f"features per epoch but model `{model_name}` expects {expected}. "
            "Make sure the uploaded file uses the same channel layout as the "
            "training data (PhysioNet EEG MMI, runs 4/8/12)."
        )
        st.stop()

y_pred = predict_epochs(model, input_data)

# Top-level metrics
n_correct = int((y_pred == y_true).sum())
n_total = len(y_true)
acc_file = n_correct / n_total if n_total else 0.0
from sklearn.metrics import f1_score
f1_file = f1_score(y_true, y_pred, average="weighted", zero_division=0) if n_total else 0.0

m1, m2, m3, _ = st.columns(4)
m1.metric("Accuracy (this file)", f"{acc_file:.3f}")
m2.metric("F1-score (this file)", f"{f1_file:.3f}")
m3.metric("Correct epochs", f"{n_correct} / {n_total}")

# Timeline
st.markdown("**Ground-truth vs Prediction timeline**")
fig_tl = plot_prediction_timeline(
    pipeline_out["starts"],
    pipeline_out["ends"],
    y_true,
    y_pred,
)
st.plotly_chart(fig_tl, use_container_width=True)

# Confusion matrix on this file
from sklearn.metrics import confusion_matrix
cm_file = confusion_matrix(y_true, y_pred, labels=[1, 2])
cmc1, cmc2 = st.columns([1, 2])
with cmc1:
    fig_cm = plot_confusion_matrix(
        cm_file,
        class_names=[CLASS_LABELS[1], CLASS_LABELS[2]],
        title=f"This file – {model_name}",
    )
    st.pyplot(fig_cm, use_container_width=True)

# Epoch-level table
with cmc2:
    df = pd.DataFrame({
        "Epoch": np.arange(1, n_total + 1),
        "Start (s)": np.round(pipeline_out["starts"], 2),
        "End (s)": np.round(pipeline_out["ends"], 2),
        "Ground Truth": [class_name(v) for v in y_true],
        "Prediction": [class_name(v) for v in y_pred],
        "Correct": ["✓" if a == b else "✗" for a, b in zip(y_true, y_pred)],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Tip: download the table or compare against the timeline above to see "
    "which time windows the model gets right or wrong."
)
