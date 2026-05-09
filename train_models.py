from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_edf_from_path
from src.features import compute_band_power_features
from src.preprocessing import apply_bandpass, extract_events, make_epochs
from src.utils import BAD_SUBJECTS, BANDS, CLASS_LABELS, FEATURE_CHANNELS, FMAX, FMIN, TARGET_RUNS, TMAX, TMIN

# ---- Configuration ----
SUBJECTS = [s for s in range(1, 41) if s not in BAD_SUBJECTS]   # S001–S040 minus bad ones
RUNS = TARGET_RUNS
DATA_PATH = Path("./data")
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./results")
RANDOM_STATE = 42

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def build_dataset(subjects: list[int], runs: list[int]):
    all_X_bp, all_X_raw, all_y = [], [], []
    per_subject: list[tuple] = []
    used_subjects = []

    for subj in subjects:
        try:
            print(f"  → Subject S{subj:03d}: downloading runs {runs}...")
            edf_paths = eegbci.load_data(subj, runs, path=str(DATA_PATH))
            raws = [load_edf_from_path(p) for p in edf_paths]
            raw = concatenate_raws(raws)

            raw_filt = apply_bandpass(raw)
            events, event_id = extract_events(raw_filt)
            if len(events) == 0:
                raise ValueError("No T1/T2 events found")

            epochs = make_epochs(raw_filt, events, event_id)
            labels = epochs.events[:, -1].astype(int)

            # ---- Two views of the same epochs: ----
            # (a) ALL channels (~64) — used for CSP and Riemannian models, which
            #     are spatial-filter methods that need full coverage to be useful.
            data_full = epochs.get_data()                       # (n_epochs, 64, n_times)

            # (b) Motor-cortex channels only (~9) — used for band-power features,
            #     where restricting to relevant channels reduces noise.
            present = [c for c in FEATURE_CHANNELS if c in epochs.ch_names]
            epochs_mc = epochs.copy().pick(present) if present else epochs
            data_mc = epochs_mc.get_data()                      # (n_epochs, 9, n_times)
            ch_names_bp = list(epochs_mc.ch_names)

            # Band-power features (9 channels)
            X_bp = compute_band_power_features(
                data_mc, sfreq=epochs.info["sfreq"], channel_names=ch_names_bp
            )

            # Raw epochs for CSP / Riemannian (64 channels), per-subject normalised
            scale = data_full.std() + 1e-10
            data_raw = (data_full - data_full.mean(axis=-1, keepdims=True)) / scale

            all_X_bp.append(X_bp)
            all_X_raw.append(data_raw)
            all_y.append(labels)
            per_subject.append((X_bp.copy(), data_raw.copy(), labels.copy()))
            used_subjects.append(subj)
            print(
                f"      kept {len(labels)} epochs "
                f"({(labels == 1).sum()} L / {(labels == 2).sum()} R)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"      ! Subject S{subj:03d} failed: {exc}")

    if not all_X_bp:
        raise RuntimeError("No subject data was successfully loaded.")

    min_ch = min(x.shape[1] for x in all_X_raw)
    all_X_raw = [x[:, :min_ch, :] for x in all_X_raw]

    X_bp = np.concatenate(all_X_bp, axis=0)
    X_raw = np.concatenate(all_X_raw, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X_bp, X_raw, y, per_subject, used_subjects


def _within_subject_eval(per_subject, use_raw, model_template):
    """Per-subject 80/20 split; aggregate test predictions across all subjects."""
    all_y_true, all_y_pred = [], []
    for X_bp_s, X_raw_s, y_s in per_subject:
        X_s = X_raw_s if use_raw else X_bp_s
        if len(y_s) < 10 or len(np.unique(y_s)) < 2:
            continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y_s, test_size=0.2, random_state=RANDOM_STATE, stratify=y_s,
            )
            m = clone(model_template)
            m.fit(X_tr, y_tr)
            all_y_true.extend(y_te.tolist())
            all_y_pred.extend(m.predict(X_te).tolist())
        except Exception:
            continue

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    report = classification_report(
        y_true, y_pred,
        target_names=[CLASS_LABELS[1], CLASS_LABELS[2]], zero_division=0,
    )
    return acc, f1, cm, report


def _cross_subject_cv(model_template, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in skf.split(X, y):
        m = clone(model_template)
        m.fit(X[tr], y[tr])
        scores.append(accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(scores)), float(np.std(scores))


def _save_result(name, acc, f1, cm, report, cv_mean, cv_std, y, model,
                 X_all, model_input_types, results, input_type):
    results[name] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_train": int(len(y) * 0.8),
        "n_test":  int(len(y) * 0.2),
        "evaluation": "within-subject 80/20 per subject, predictions aggregated",
    }
    model_input_types[name] = input_type
    model.fit(X_all, y)
    joblib.dump(model, MODELS_DIR / f"{name}.pkl")
    print(f"    saved → {MODELS_DIR / f'{name}.pkl'}")


def train_and_evaluate(X_bp, X_raw, y, per_subject):
    # ---- Band-power models (StandardScaler handles scaling internally) ----
    bp_models = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=None,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                solver="adam", max_iter=1000, early_stopping=True,
                validation_fraction=0.15, random_state=RANDOM_STATE,
                alpha=1e-3,   # stronger L2 to fight overfitting on small data
            )),
        ]),
    }

    # ---- CSP model (now uses ALL 64 channels — spatial filters need full coverage) ----
    csp_models = {
        "csp_lda": Pipeline([
            ("csp", CSP(n_components=6, reg="ledoit_wolf", log=True, norm_trace=True)),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    }

    # ---- Riemannian geometry models (best for small EEG samples) ----
    # Covariances(lwf) = per-epoch Ledoit-Wolf covariance → 9×9 SPD matrix
    # MDM classifies by geodesic distance to the Riemannian mean of each class
    # TangentSpace maps to the tangent plane at the Fréchet mean then uses LDA
    riemann_models = {
        "riemann_mdm": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("clf", MDM(metric="riemann")),
        ]),
        "riemann_ts_lda": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("ts",  TangentSpace(metric="riemann")),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    }

    results: dict = {}
    model_input_types: dict = {}

    for name, model in bp_models.items():
        print(f"\n  Evaluating {name} (band-power, within-subject) ...")
        acc, f1, cm, report = _within_subject_eval(per_subject, False, model)
        cv_mean, cv_std = _cross_subject_cv(model, X_bp, y)
        print(f"    within-subj acc={acc:.3f}  f1={f1:.3f}  cross-subj CV={cv_mean:.3f}±{cv_std:.3f}")
        print(f"  Training {name} on all data ...")
        _save_result(name, acc, f1, cm, report, cv_mean, cv_std,
                     y, model, X_bp, model_input_types, results, "band_power")

    for name, model in {**csp_models, **riemann_models}.items():
        print(f"\n  Evaluating {name} (raw epochs, within-subject) ...")
        acc, f1, cm, report = _within_subject_eval(per_subject, True, model)
        cv_mean, cv_std = _cross_subject_cv(model, X_raw, y)
        print(f"    within-subj acc={acc:.3f}  f1={f1:.3f}  cross-subj CV={cv_mean:.3f}±{cv_std:.3f}")
        print(f"  Training {name} on all data ...")
        _save_result(name, acc, f1, cm, report, cv_mean, cv_std,
                     y, model, X_raw, model_input_types, results, "raw_epochs")

    return results, model_input_types


def main():
    print("=" * 70)
    print("  PhysioNet EEG MMI — offline model training")
    print("=" * 70)

    print(f"\nBuilding dataset  subjects {SUBJECTS[0]}–{SUBJECTS[-1]}  runs {RUNS} ...")
    X_bp, X_raw, y, per_subject, used_subjects = build_dataset(SUBJECTS, RUNS)
    print(f"\nX_bp={X_bp.shape}  X_raw={X_raw.shape}  y={y.shape}")
    print(f"Class balance: left={int((y==1).sum())}  right={int((y==2).sum())}")

    print("\nTraining and evaluating models ...")
    results, model_input_types = train_and_evaluate(X_bp, X_raw, y, per_subject)

    metadata = {
        "training_subjects": used_subjects,
        "runs": RUNS,
        "preprocessing": {
            "bandpass_low_hz": FMIN, "bandpass_high_hz": FMAX,
            "filter_design": "FIR (firwin)",
            "epoch_tmin_s": TMIN, "epoch_tmax_s": TMAX,
            "channels": (
                f"Band-power: 9 motor-cortex channels {FEATURE_CHANNELS}; "
                f"CSP/Riemannian: ALL 64 EEG channels (spatial filtering)"
            ),
        },
        "feature_extraction": {
            "band_power": "9-channel Welch PSD → log mean power per channel × band",
            "csp":        "64-channel CSP log-variance (6 components, Ledoit-Wolf reg)",
            "riemannian": "64-channel Ledoit-Wolf covariance per epoch (SPD matrix)",
            "bands_hz": {k: list(v) for k, v in BANDS.items()},
            "n_features_per_epoch": int(X_bp.shape[1]),
        },
        "evaluation_strategy": (
            "Within-subject 80/20 split per subject, predictions aggregated. "
            "Reflects individual-level performance (standard BCI evaluation). "
            "Cross-subject 5-fold CV also reported."
        ),
        "model_input_types": model_input_types,
        "results": results,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata → {MODELS_DIR / 'model_metadata.json'}")

    rows = []
    for name, res in results.items():
        rows.append({
            "model": name,
            "input": model_input_types[name],
            "within_subject_accuracy": res["accuracy"],
            "within_subject_f1":       res["f1_score"],
            "cross_subject_cv_mean":   res["cv_accuracy_mean"],
            "cross_subject_cv_std":    res["cv_accuracy_std"],
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "offline_model_results.csv", index=False)
    print(f"✓ CSV → {RESULTS_DIR / 'offline_model_results.csv'}")
    print("\nDone.  streamlit run app.py")


if __name__ == "__main__":
    main()
