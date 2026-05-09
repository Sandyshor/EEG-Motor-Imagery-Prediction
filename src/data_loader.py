"""Load EDF files and extract basic metadata.

The PhysioNet EEG Motor Movement/Imagery dataset uses channel names like
``Fc5.``, ``C3..``, etc. ``mne.datasets.eegbci.standardize`` rewrites these to
standard 10-05 names (``FC5``, ``C3``, ...). We always run it on load so that
training and inference share the exact same channel naming.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import mne
from mne.datasets import eegbci
from mne.io import Raw, read_raw_edf


def load_edf_from_path(path: str | Path, preload: bool = True) -> Raw:
    """Read an EDF file from disk and standardise channel names."""
    raw = read_raw_edf(str(path), preload=preload, verbose=False)
    eegbci.standardize(raw)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        # If montage cannot be applied (unusual channel names) just continue;
        # it only affects topographical plots, not classification.
        pass
    return raw


def load_edf_from_upload(uploaded_file) -> Raw:
    """Read a Streamlit ``UploadedFile`` by writing it to a temporary path.

    MNE's EDF reader needs a real file on disk, so we write the bytes to a
    temporary file first and let the OS clean it up later.
    """
    suffix = Path(uploaded_file.name).suffix or ".edf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    return load_edf_from_path(tmp_path)


def get_file_info(raw: Raw) -> dict[str, Any]:
    """Return a small dict of summary information for the UI."""
    annotations = raw.annotations
    unique_descs = sorted(set(annotations.description)) if len(annotations) else []
    return {
        "sampling_frequency": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
        "channel_names": raw.ch_names,
        "duration_seconds": float(raw.times[-1]) if len(raw.times) else 0.0,
        "n_annotations": len(annotations),
        "annotation_descriptions": unique_descs,
    }
