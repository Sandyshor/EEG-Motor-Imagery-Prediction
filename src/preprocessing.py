"""Preprocessing: band-pass filtering and event-based epoch creation.

Both training and inference must use identical preprocessing, so all the
parameters live in :mod:`src.utils` and are imported here.
"""
from __future__ import annotations

import numpy as np
import mne
from mne.io import Raw

from .utils import EVENT_ID, FEATURE_CHANNELS, FMAX, FMIN, TMAX, TMIN


def apply_bandpass(raw: Raw, fmin: float = FMIN, fmax: float = FMAX) -> Raw:
    """Apply a zero-phase FIR band-pass filter in-place and return the Raw.

    For motor imagery, mu (8–13 Hz) and beta (13–30 Hz) rhythms over the
    sensorimotor cortex carry the discriminative information, so an 8–30 Hz
    band-pass is a standard choice.
    """
    raw_filtered = raw.copy().filter(
        l_freq=fmin,
        h_freq=fmax,
        fir_design="firwin",
        skip_by_annotation="edge",
        verbose=False,
    )
    return raw_filtered


def extract_events(raw: Raw) -> tuple[np.ndarray, dict[str, int]]:
    """Extract T1/T2 events from the EDF annotations.

    Returns
    -------
    events : ndarray of shape (n_events, 3)
        Standard MNE events array (sample, prev_id, event_id).
    event_id : dict
        ``{"T1": 1, "T2": 2}`` (only the events actually present).
    """
    events, event_id = mne.events_from_annotations(
        raw, event_id=EVENT_ID, verbose=False
    )
    return events, event_id


def make_epochs(
    raw: Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    tmin: float = TMIN,
    tmax: float = TMAX,
) -> mne.Epochs:
    """Create event-based epochs around T1/T2 events on the (already filtered) Raw."""
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=False,
        picks="eeg",
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


def preprocess_raw_to_epochs(raw: Raw) -> tuple[mne.Epochs, np.ndarray]:
    """Full pipeline: band-pass filter, extract T1/T2 events, build epochs.

    Returns
    -------
    epochs : mne.Epochs
    labels : ndarray of int (1 = left, 2 = right)
    """
    raw_filt = apply_bandpass(raw)
    events, event_id = extract_events(raw_filt)
    if len(events) == 0:
        raise ValueError(
            "No T1/T2 events found in this file. "
            "Please upload an EDF from runs 4, 8, or 12 of the PhysioNet "
            "EEG Motor Movement/Imagery dataset."
        )
    epochs = make_epochs(raw_filt, events, event_id)

    # Restrict to motor-cortex channels (if specified and present in the file).
    # Same selection is used at training and inference time, so feature
    # dimensions stay consistent.
    if FEATURE_CHANNELS:
        present = [c for c in FEATURE_CHANNELS if c in epochs.ch_names]
        if present:
            epochs = epochs.pick(present)

    labels = epochs.events[:, -1].astype(int)
    return epochs, labels
