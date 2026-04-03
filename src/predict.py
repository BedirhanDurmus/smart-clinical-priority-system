"""Inference helpers for the trained model."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import issparse

from .preprocess import get_feature_columns, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .chief_complaint import normalise_complaint, categorise_complaint

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

KTAS_LABELS = {
    1: "Resuscitation (Level 1 – Immediate)",
    2: "Emergency (Level 2 – Very Urgent)",
    3: "Urgent (Level 3)",
    4: "Less Urgent (Level 4)",
    5: "Non-Urgent (Level 5)",
}


def _ensure_preprocess_pickling_compat() -> None:
    """
    The pickled Pipeline in best_model.joblib wires FunctionTransformer to
    src.preprocess._column_to_string_list / _tfidf_to_dense.
    Older preprocess modules can make joblib.load raise AttributeError; we patch missing names here.
    """
    import src.preprocess as pre

    if not hasattr(pre, "_column_to_string_list"):

        def _column_to_string_list(X):
            flat = np.asarray(X).ravel()
            return np.array(
                [str(x) if x is not None and str(x) != "nan" else "unknown" for x in flat]
            )

        pre._column_to_string_list = _column_to_string_list

    if not hasattr(pre, "_tfidf_to_dense"):

        def _tfidf_to_dense(X):
            if issparse(X):
                return X.toarray()
            return np.asarray(X)

        pre._tfidf_to_dense = _tfidf_to_dense


def load_model(path: Path | str | None = None):
    path = Path(path) if path else MODELS_DIR / "best_model.joblib"
    _ensure_preprocess_pickling_compat()
    return joblib.load(path)


def predict_single(
    model,
    age: float,
    sex: int,
    group: int,
    arrival_mode: int,
    injury: int,
    chief_complaint: str,
    mental: int,
    pain: int,
    nrs_pain: float | None,
    sbp: float | None,
    dbp: float | None,
    hr: float | None,
    rr: float | None,
    bt: float | None,
    saturation: float | None,
    patients_per_hour: int = 5,
) -> dict:
    """Build a single-row DataFrame and return prediction + probabilities."""
    cc_norm = normalise_complaint(chief_complaint)
    cc_cat = categorise_complaint(cc_norm)

    row = {
        "Age": age,
        "Patients number per hour": patients_per_hour,
        "SBP": sbp,
        "DBP": dbp,
        "HR": hr,
        "RR": rr,
        "BT": bt,
        "Saturation": saturation,
        "Group": str(group),
        "Sex": str(sex),
        "Arrival mode": str(arrival_mode),
        "Injury": str(injury),
        "Mental": str(mental),
        "Pain": str(pain),
        "cc_category": cc_cat,
        "NRS_pain": nrs_pain,
        "cc_normalised": cc_norm,
    }

    feature_cols = get_feature_columns()
    X = pd.DataFrame([row], columns=feature_cols)

    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]

    return {
        "predicted_ktas": pred,
        "label": KTAS_LABELS.get(pred, "Unknown"),
        "probabilities": {int(c): round(float(p), 4) for c, p in zip(model.classes_, proba)} if proba is not None else None,
        "cc_normalised": cc_norm,
        "cc_category": cc_cat,
    }
