"""Plotting helpers for the Streamlit app."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.signal import welch

from .utils import CLASS_LABELS


# ---------- Signal plots ----------

def plot_raw_signal(
    times: np.ndarray,
    signal: np.ndarray,
    channel_name: str,
    events: np.ndarray | None = None,
    sfreq: float | None = None,
):
    """Plot a single-channel raw EEG trace with optional event markers."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, signal * 1e6, linewidth=0.6, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Raw EEG — channel {channel_name}")
    ax.grid(alpha=0.3)

    if events is not None and sfreq is not None and len(events):
        colours = {1: "tab:green", 2: "tab:red"}
        labels_seen = set()
        for sample, _, eid in events:
            t = sample / sfreq
            if not (times[0] <= t <= times[-1]):
                continue
            label = f"T{eid}"
            ax.axvline(
                t,
                color=colours.get(int(eid), "gray"),
                linestyle="--",
                alpha=0.6,
                label=label if label not in labels_seen else None,
            )
            labels_seen.add(label)
        if labels_seen:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_raw_vs_filtered(
    times: np.ndarray,
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    channel_name: str,
    fmin: float,
    fmax: float,
):
    """Overlay raw vs band-pass filtered signal for visual comparison."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, raw_signal * 1e6, alpha=0.5, label="Raw", color="#1f77b4", linewidth=0.6)
    ax.plot(
        times, filtered_signal * 1e6,
        label=f"Filtered ({fmin:g}–{fmax:g} Hz)",
        color="#d62728", linewidth=0.7,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Raw vs Filtered — channel {channel_name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_psd(signal: np.ndarray, sfreq: float, channel_name: str, fmax: float = 60):
    """Welch power spectral density on a log scale."""
    nperseg = min(int(sfreq * 2), len(signal))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)
    mask = freqs <= fmax

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(freqs[mask], psd[mask], color="#9467bd")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (V²/Hz)")
    ax.set_title(f"PSD — channel {channel_name}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_band_power_by_class(features: np.ndarray, labels: np.ndarray, n_bands: int):
    """Bar chart: mean band power per band, averaged across channels and epochs,
    split by class. Used as a sanity check for the feature extraction step."""
    n_features = features.shape[1]
    n_channels = n_features // n_bands

    # Reshape back to (n_epochs, n_channels, n_bands)
    reshaped = features.reshape(-1, n_channels, n_bands)
    band_names = ["Mu/Alpha (8–13 Hz)", "Beta (13–30 Hz)"][:n_bands]

    means = {}
    for cls in np.unique(labels):
        # mean across epochs and channels
        means[int(cls)] = reshaped[labels == cls].mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(band_names))
    width = 0.35
    colours = {1: "#2ca02c", 2: "#d62728"}
    for i, (cls, vals) in enumerate(means.items()):
        ax.bar(
            x + (i - 0.5) * width, vals * 1e12, width,
            label=CLASS_LABELS.get(cls, str(cls)),
            color=colours.get(cls, "gray"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Mean band power (µV²/Hz)")
    ax.set_title("Mean band power by class (averaged across channels)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------- Confusion matrix ----------

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix"):
    """Display a confusion matrix with annotated counts."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ---------- Prediction timeline (Plotly, interactive) ----------

def plot_prediction_timeline(
    epoch_starts: np.ndarray,
    epoch_ends: np.ndarray,
    ground_truth: np.ndarray,
    predictions: np.ndarray,
):
    """Two stacked horizontal bar timelines: ground truth on top, predictions below."""
    colour_map = {1: "#2ca02c", 2: "#d62728"}

    fig = go.Figure()

    for label, gt_pred, y in [("Ground Truth", ground_truth, 1), ("Prediction", predictions, 0)]:
        for start, end, cls in zip(epoch_starts, epoch_ends, gt_pred):
            fig.add_trace(go.Bar(
                x=[end - start],
                y=[label],
                base=[start],
                orientation="h",
                marker=dict(color=colour_map.get(int(cls), "gray")),
                hovertemplate=(
                    f"{label}<br>"
                    f"Class: {CLASS_LABELS.get(int(cls), cls)}<br>"
                    f"Start: {start:.2f} s<br>End: {end:.2f} s<extra></extra>"
                ),
                showlegend=False,
            ))

    # Dummy traces for legend
    for cls, name in CLASS_LABELS.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation="h",
            marker=dict(color=colour_map[cls]),
            name=name, showlegend=True,
        ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="Time (s)",
        height=220,
        margin=dict(l=20, r=20, t=30, b=40),
        title="Ground Truth vs Prediction Timeline",
        legend=dict(orientation="h", y=-0.3),
    )
    return fig
