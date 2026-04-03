"""Feature engineering and preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from scipy.sparse import issparse

from .chief_complaint import add_complaint_features


def _tfidf_to_dense(X):
    """Convert sparse TF-IDF output to dense for stacking with other numeric blocks."""
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _column_to_string_list(X):
    """ColumnTransformer passes (n_samples, 1); TfidfVectorizer needs 1d strings."""
    flat = np.asarray(X).ravel()
    return np.array([str(x) if x is not None and str(x) != "nan" else "unknown" for x in flat])

# Sentinel values that represent missing data in the raw CSV
_MISSING_SENTINELS = {"??", "#BO\ufffd!", "#BO\xde!", ""}

# Columns that would leak the target
LEAK_COLUMNS = [
    "Diagnosis in ED",
    "Disposition",
    "KTAS_RN",
    "Error_group",
    "mistriage",
    "Length of stay_min",
    "KTAS duration_min",
]

TARGET = "KTAS_expert"

NUMERIC_FEATURES = ["Age", "Patients number per hour", "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
CATEGORICAL_FEATURES = ["Group", "Sex", "Arrival mode", "Injury", "Mental", "Pain"]
TEXT_FEATURE = "cc_category"  # derived from Chief_complain
CC_TEXT_FEATURE = "cc_normalised"  # for TF-IDF (same pipeline as add_complaint_features)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce sentinel strings to NaN in numeric columns."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({s: np.nan for s in _MISSING_SENTINELS})
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the target column is integer 1-5."""
    df = df.copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].between(1, 5)]
    return df


def clean_nrs_pain(df: pd.DataFrame) -> pd.DataFrame:
    """Handle NRS_pain: numeric where valid, NaN otherwise."""
    df = df.copy()
    df["NRS_pain"] = pd.to_numeric(
        df["NRS_pain"].astype(str).str.strip().replace({s: np.nan for s in _MISSING_SENTINELS}),
        errors="coerce",
    )
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: complaint features, numeric cleaning, target cleaning."""
    df = add_complaint_features(df)
    df = clean_numeric(df)
    df = clean_nrs_pain(df)
    df = clean_target(df)

    # Cast categorical features to string to avoid downstream issues
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature columns used by the model."""
    return (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TEXT_FEATURE, "NRS_pain", CC_TEXT_FEATURE]
    )


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the feature set."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    cc_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    nrs_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # TF-IDF on normalised free text — captures wording beyond coarse category
    tfidf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        (
            "as_1d",
            FunctionTransformer(_column_to_string_list, validate=False),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=200,
                min_df=2,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        ("dense", FunctionTransformer(_tfidf_to_dense, validate=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("cc", cc_pipe, [TEXT_FEATURE]),
            ("nrs", nrs_pipe, ["NRS_pain"]),
            ("txt", tfidf_pipe, [CC_TEXT_FEATURE]),
        ],
        remainder="drop",
    )
    return preprocessor
