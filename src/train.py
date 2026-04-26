"""Model training, evaluation, hyperparameter search, and persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    make_scorer,
)
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency at runtime
    LGBMClassifier = None

from .preprocess import (
    build_preprocessor,
    get_feature_columns,
    TARGET,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RANDOM_STATE = 42


class LabelShiftWrapper(BaseEstimator, ClassifierMixin):
    """Shift labels from 1-based to 0-based for XGBoost."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        self.offset_ = int(np.min(y))
        self.estimator.fit(X, y - self.offset_, **kwargs)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.estimator.predict(X) + self.offset_

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def _get_X_y(df: pd.DataFrame):
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df[TARGET].values
    return X, y


def build_pipelines(preprocessor) -> dict[str, Pipeline]:
    """Baseline pipelines with regularisation-oriented defaults (reduces overfitting)."""
    pipelines = {
        "LogisticRegression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=3000,
                C=0.3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "RandomForest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=8,
                min_samples_split=12,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("pre", preprocessor),
            ("clf", LabelShiftWrapper(XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=5.0,
                gamma=0.2,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ))),
        ]),
    }
    if LGBMClassifier is not None:
        pipelines["LightGBM"] = Pipeline([
            ("pre", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=25,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.3,
                reg_lambda=5.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ])
    return pipelines


def evaluate_cv(pipeline: Pipeline, X, y, cv=5) -> dict:
    """Stratified cross-val (out-of-fold predictions)."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    return {
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "f1_macro": round(f1_score(y, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y, y_pred, average="weighted"), 4),
        "report": classification_report(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "y_pred": y_pred,
    }


def evaluate_train_vs_test(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """Fit once on train, report train vs test metrics (overfitting gap)."""
    pipe = clone(pipeline)
    pipe.fit(X_train, y_train)
    y_tr = pipe.predict(X_train)
    y_te = pipe.predict(X_test)
    return {
        "train_accuracy": round(accuracy_score(y_train, y_tr), 4),
        "test_accuracy": round(accuracy_score(y_test, y_te), 4),
        "train_f1_macro": round(f1_score(y_train, y_tr, average="macro"), 4),
        "test_f1_macro": round(f1_score(y_test, y_te, average="macro"), 4),
        "train_f1_weighted": round(f1_score(y_train, y_tr, average="weighted"), 4),
        "test_f1_weighted": round(f1_score(y_test, y_te, average="weighted"), 4),
        "gap_f1_macro": round(
            f1_score(y_train, y_tr, average="macro")
            - f1_score(y_test, y_te, average="macro"),
            4,
        ),
        "confusion_matrix_test": confusion_matrix(y_test, y_te),
        "classification_report_test": classification_report(y_test, y_te, zero_division=0),
    }


def randomized_search_lightgbm(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = 45,
    cv: int = 5,
) -> RandomizedSearchCV:
    """Tune LightGBM with regularisation-heavy search space (generalisation focus)."""
    if LGBMClassifier is None:
        raise ImportError(
            "lightgbm is not installed. Install it with `pip install lightgbm` "
            "to run LightGBM search/training."
        )
    base = LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    pipe = Pipeline([("pre", preprocessor), ("clf", base)])

    # Conservative ranges to limit overfitting (small ED dataset)
    param_distributions = {
        "clf__n_estimators": [80, 120, 180, 250],
        "clf__learning_rate": [0.02, 0.04, 0.06, 0.08],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__num_leaves": [15, 31, 45],
        "clf__min_child_samples": [25, 40, 60, 90],
        "clf__subsample": [0.65, 0.75, 0.85],
        "clf__colsample_bytree": [0.65, 0.75, 0.85],
        "clf__reg_alpha": [0.1, 0.5, 1.0, 2.0],
        "clf__reg_lambda": [5.0, 10.0, 15.0, 25.0],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = make_scorer(f1_score, average="macro")

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=skf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def train_and_save(df: pd.DataFrame, model_name: str = "LightGBM") -> Path:
    """Train a named baseline pipeline on full data and save."""
    preprocessor = build_preprocessor()
    pipelines = build_pipelines(preprocessor)
    pipe = pipelines[model_name]
    X, y = _get_X_y(df)
    pipe.fit(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(pipe, model_path)
    return model_path


def train_and_save_pipeline(pipe: Pipeline, path: Path | None = None) -> Path:
    """Save an arbitrary fitted pipeline (e.g. from RandomizedSearchCV.best_estimator_)."""
    MODELS_DIR.mkdir(exist_ok=True)
    out = path or (MODELS_DIR / "best_model.joblib")
    joblib.dump(pipe, out)
    return out


def run_improved_training(
    df: pd.DataFrame,
    test_size: float = 0.2,
    search_n_iter: int = 45,
) -> dict:
    """
    1) Stratified hold-out test set (honest generalisation).
    2) RandomizedSearchCV on train (macro-F1).
    3) Compare CV best score vs hold-out test (overfitting check).
    4) Refit best pipeline on ALL rows and save for deployment.
    """
    X, y = _get_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()

    # --- Baselines on same split (quick comparison) ---
    baselines = build_pipelines(preprocessor)
    baseline_rows = []
    for name, pipe in baselines.items():
        m = evaluate_train_vs_test(pipe, X_train, y_train, X_test, y_test)
        baseline_rows.append({"model": name, **m})
        print(f"[baseline] {name}: test F1-macro={m['test_f1_macro']}, gap={m['gap_f1_macro']}")

    # --- Tuned LightGBM ---
    print("\nRunning RandomizedSearchCV (LightGBM)...")
    search = randomized_search_lightgbm(
        preprocessor, X_train, y_train, n_iter=search_n_iter, cv=5,
    )
    best_cv = search.best_score_
    print(f"Best CV macro-F1: {best_cv:.4f}")
    print(f"Best params: {search.best_params_}")

    tuned_metrics = evaluate_train_vs_test(
        search.best_estimator_, X_train, y_train, X_test, y_test,
    )
    print(
        f"Tuned LightGBM — train F1-macro={tuned_metrics['train_f1_macro']}, "
        f"test F1-macro={tuned_metrics['test_f1_macro']}, gap={tuned_metrics['gap_f1_macro']}"
    )

    # Pick model with best hold-out F1 among candidates with acceptable generalisation gap
    candidates: list[tuple[str, Pipeline, dict]] = [
        ("TunedLightGBM", search.best_estimator_, tuned_metrics),
    ]
    for row in baseline_rows:
        name = row["model"]
        pipe = baselines[name]
        m = {k: row[k] for k in row if k != "model"}
        candidates.append((name, pipe, m))

    GAP_THRESHOLD = 0.18
    acceptable = [c for c in candidates if c[2]["gap_f1_macro"] <= GAP_THRESHOLD]
    if acceptable:
        chosen_name, chosen_pipe, chosen_metrics = max(
            acceptable, key=lambda x: x[2]["test_f1_macro"],
        )
    else:
        chosen_name, chosen_pipe, chosen_metrics = min(
            candidates, key=lambda x: x[2]["gap_f1_macro"],
        )
    print(
        f"\nSelected for deployment: {chosen_name} "
        f"(hold-out F1-macro={chosen_metrics['test_f1_macro']}, gap={chosen_metrics['gap_f1_macro']})"
    )

    # Refit chosen pipeline on full data for production
    full_pipe = clone(chosen_pipe)
    full_pipe.fit(X, y)

    model_path = train_and_save_pipeline(full_pipe)

    return {
        "baseline_table": baseline_rows,
        "best_cv_macro_f1": float(best_cv),
        "best_lgbm_params": search.best_params_,
        "tuned_lgbm_holdout": tuned_metrics,
        "chosen_model": chosen_name,
        "chosen_holdout": chosen_metrics,
        "model_path": str(model_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


__all__ = [
    "LabelShiftWrapper",
    "_get_X_y",
    "build_pipelines",
    "evaluate_cv",
    "evaluate_train_vs_test",
    "randomized_search_lightgbm",
    "train_and_save",
    "train_and_save_pipeline",
    "run_improved_training",
]
