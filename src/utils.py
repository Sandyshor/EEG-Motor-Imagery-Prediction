"""Shared constants and small helpers used across the pipeline."""

# ---- Preprocessing / feature settings (must match training and inference) ----
FMIN = 8.0          # band-pass low cutoff (Hz)
FMAX = 30.0         # band-pass high cutoff (Hz)
TMIN = 0.5          # epoch start relative to event (s) — start AFTER cue
TMAX = 2.5          # epoch end relative to event (s) — focus on imagery period

# Frequency bands for band-power features
BANDS = {
    "mu_alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Subjects with known data-quality issues in PhysioNet EEG MMI
# (different sampling rates, missing/corrupt annotations, etc.)
BAD_SUBJECTS = [88, 89, 92, 100, 104]

# Event ID mapping used for training and inference
# In PhysioNet runs 4, 8, 12: T1 = imagined left fist, T2 = imagined right fist
EVENT_ID = {"T1": 1, "T2": 2}

# Human-readable class labels (keys must match EVENT_ID values)
CLASS_LABELS = {
    1: "Left fist (T1)",
    2: "Right fist (T2)",
}

# Motor-cortex channels often highlighted for motor-imagery tasks
MOTOR_CHANNELS = ["C3", "Cz", "C4"]

# Channels actually used as feature inputs. Motor imagery produces
# event-related desynchronisation over the sensorimotor cortex, so
# restricting to these channels reduces noise from the rest of the scalp
# and substantially improves cross-subject accuracy.
# Set to None to fall back to all EEG channels.
FEATURE_CHANNELS = [
    "FC3", "FCz", "FC4",
    "C3",  "Cz",  "C4",
    "CP3", "CPz", "CP4",
]

# Runs that contain the imagined left/right fist task in PhysioNet EEG MMI
TARGET_RUNS = [4, 8, 12]


def class_name(label: int) -> str:
    """Return a human-readable class name for a numeric label."""
    return CLASS_LABELS.get(int(label), f"Unknown ({label})")
