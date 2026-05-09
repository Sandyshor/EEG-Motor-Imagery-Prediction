"""Feature extraction.

For each epoch we compute the mean band power in the mu (8–13 Hz) and beta
(13–30 Hz) bands, per EEG channel, using Welch's method. The resulting feature
vector has length ``n_channels * n_bands`` per epoch.

Why band power? Motor imagery produces event-related desynchronisation /
synchronisation in the mu and beta rhythms over the sensorimotor cortex,
which manifests as a power change in those bands. Mean band power per
channel is a simple, well-justified, and reproducible feature for this task.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .utils import BANDS


def compute_band_power_features(
    epoch_data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] = BANDS,
    channel_names: list[str] | None = None,
) -> np.ndarray:
    """Compute mean band power per (channel, band) for each epoch.

    Parameters
    ----------
    epoch_data : ndarray of shape (n_epochs, n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz.
    bands : dict
        Mapping of band name to ``(fmin, fmax)`` in Hz.
    channel_names : list of str, optional
        When provided, appends C3/C4 laterality index features — the primary
        biomarker for left vs. right hand motor imagery.

    Returns
    -------
    features : ndarray of shape (n_epochs, n_channels * n_bands [+ n_bands])
    """
    if epoch_data.ndim != 3:
        raise ValueError(
            f"Expected 3D epoch data (n_epochs, n_channels, n_samples); "
            f"got shape {epoch_data.shape}"
        )

    n_samples = epoch_data.shape[-1]
    nperseg = min(256, n_samples)
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd shape: (n_epochs, n_channels, n_freqs)

    band_powers = []
    for _, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        # Mean power within the band
        band_power = psd[..., mask].mean(axis=-1)  # (n_epochs, n_channels)
        band_powers.append(band_power)

    # Stack along channel axis → (n_epochs, n_channels, n_bands), then flatten
    stacked = np.stack(band_powers, axis=-1)
    features = stacked.reshape(stacked.shape[0], -1)

    # Band power is approximately log-normally distributed; the log transform
    # gives classifiers (especially linear ones / MLPs) a much easier time.
    features = np.log(features + 1e-20)

    # C3/C4 laterality index (C4−C3)/(C4+C3) per band.
    # C3 is over the left motor cortex (controls right hand) and C4 over the
    # right (controls left hand), so this ratio is the most discriminative
    # single feature for left vs. right hand motor imagery.
    if channel_names is not None:
        ch_upper = [ch.upper() for ch in channel_names]
        c3_idx = ch_upper.index("C3") if "C3" in ch_upper else None
        c4_idx = ch_upper.index("C4") if "C4" in ch_upper else None
        if c3_idx is not None and c4_idx is not None:
            lat_cols = []
            for _, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                c3_pw = psd[:, c3_idx, :][:, mask].mean(axis=-1)
                c4_pw = psd[:, c4_idx, :][:, mask].mean(axis=-1)
                lat = (c4_pw - c3_pw) / (c4_pw + c3_pw + 1e-20)
                lat_cols.append(lat[:, np.newaxis])
            features = np.hstack([features] + lat_cols)

    return features


def feature_names(channel_names: list[str], bands: dict = BANDS) -> list[str]:
    """Return feature column names matching the order produced above."""
    names = []
    for ch in channel_names:
        for band in bands.keys():
            names.append(f"{ch}_{band}")
    return names
