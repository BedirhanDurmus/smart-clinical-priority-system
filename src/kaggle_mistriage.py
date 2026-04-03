"""
Helpers aligned with `triage-application-with-machine-learning-models.ipynb` for
mistriage (normal / over / under triage) preprocessing and prediction.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.load_data import load_raw

MISTRIAGE_LABELS = ["Normal Triage", "Over Triage", "Under Triage"]


def _replace_bad_nrs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "NRS_pain" in df.columns:
        s = df["NRS_pain"].astype(str)
        bad = s.str.contains(r"^#", na=False) | s.str.lower().isin(["nan", "nat", ""])
        df.loc[bad, "NRS_pain"] = np.nan
        df["NRS_pain"] = pd.to_numeric(df["NRS_pain"], errors="coerce")
    return df


def _fill_missing_notebook(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["NRS_pain", "Saturation", "Diagnosis in ED"]:
        if col not in df.columns:
            continue
        df[col] = df.groupby(["mistriage", "KTAS_expert"])[col].transform(
            lambda x: x.fillna(x.mode()[0]) if len(x.mode()) else x
        )
    return df


def _numeric_to_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [
        "Group",
        "Sex",
        "Arrival mode",
        "Injury",
        "Pain",
        "Mental",
        "Disposition",
        "KTAS_RN",
        "KTAS_expert",
        "mistriage",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(object)
    injury_cat = ["No", "Yes"]
    sex_cat = ["Female", "Male"]
    pain_cat = ["No", "Yes"]
    mental_cat = ["Alert", "Verbose Response", "Pain Response", "Unresponsive"]
    group_cat = ["Local ED (3th Degree)", "Regional ED (4th Degree)"]
    arrival_mode_cat = [
        "Walking",
        "Public Ambulance",
        "Private Vehicle",
        "Private Ambulance",
        "Other",
        "Other",
        "Other",
    ]
    disposition_cat = [
        "Discharge",
        "Admission to Ward",
        "Admission to ICU",
        "Discharge",
        "Transfer",
        "Death",
        "Surgery",
    ]
    KTAS_cat = ["Emergency", "Emergency", "Emergency", "Non-Emergency", "Non-Emergency"]

    if "KTAS duration_min" in df.columns:
        kdur = pd.to_numeric(
            df["KTAS duration_min"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df["KTAS duration_min"] = kdur.apply(lambda x: int(round(x)) if pd.notna(x) else x)
    if "NRS_pain" in df.columns:
        df["NRS_pain"] = df["NRS_pain"].apply(
            lambda x: int(round(float(x))) if pd.notna(x) and str(x).strip() != "" else x
        )

    df.loc[df["Sex"] == 1, "Sex"] = sex_cat[0]
    df.loc[df["Sex"] == 2, "Sex"] = sex_cat[1]
    df.loc[df["Injury"] == 1, "Injury"] = injury_cat[0]
    df.loc[df["Injury"] == 2, "Injury"] = injury_cat[1]
    df.loc[df["Pain"] == 0, "Pain"] = pain_cat[0]
    df.loc[df["Pain"] == 1, "Pain"] = pain_cat[1]
    df.loc[df["Mental"] == 1, "Mental"] = mental_cat[0]
    df.loc[df["Mental"] == 2, "Mental"] = mental_cat[1]
    df.loc[df["Mental"] == 3, "Mental"] = mental_cat[2]
    df.loc[df["Mental"] == 4, "Mental"] = mental_cat[3]
    df.loc[df["Group"] == 1, "Group"] = group_cat[0]
    df.loc[df["Group"] == 2, "Group"] = group_cat[1]
    for i in range(1, 8):
        df.loc[df["Arrival mode"] == i, "Arrival mode"] = arrival_mode_cat[i - 1]
    for i in range(1, 8):
        df.loc[df["Disposition"] == i, "Disposition"] = disposition_cat[i - 1]
    for i in range(1, 6):
        df.loc[df["KTAS_RN"] == i, "KTAS_RN"] = KTAS_cat[i - 1]
        df.loc[df["KTAS_expert"] == i, "KTAS_expert"] = KTAS_cat[i - 1]

    mistriage_cat = MISTRIAGE_LABELS
    df.loc[df["mistriage"] == 0, "mistriage"] = mistriage_cat[0]
    df.loc[df["mistriage"] == 1, "mistriage"] = mistriage_cat[1]
    df.loc[df["mistriage"] == 2, "mistriage"] = mistriage_cat[2]

    vitals = ["SBP", "DBP", "HR", "RR", "BT", "Saturation"]
    for c in vitals:
        if c not in df.columns:
            continue
        df[c] = df[c].replace("??", np.nan)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "SBP" in df.columns:
        m = df["SBP"].replace(0, np.nan).mode()
        fill_sbp = m.iloc[0] if len(m) else df["SBP"].median()
        df["SBP"] = df["SBP"].replace(0, fill_sbp)
    if "DBP" in df.columns:
        m = df["DBP"].replace(0, np.nan).mode()
        fill_dbp = m.iloc[0] if len(m) else df["DBP"].median()
        df["DBP"] = df["DBP"].replace(0, fill_dbp)

    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    interval = (0, 25, 45, 60, 100)
    cats = ["Young", "Adult", "Mid_Age", "Old"]
    df["New_Age"] = pd.cut(df["Age"], interval, labels=cats)

    df.loc[df["SBP"] < 80, "New_SBP"] = "Low"
    df.loc[(df["SBP"] >= 80) & (df["SBP"] <= 120), "New_SBP"] = "Normal"
    df.loc[df["SBP"] > 120, "New_SBP"] = "High"

    df.loc[df["DBP"] < 60, "New_DBP"] = "Low"
    df.loc[(df["DBP"] >= 60) & (df["DBP"] <= 80), "New_DBP"] = "Normal"
    df.loc[df["DBP"] > 80, "New_DBP"] = "High"

    df.loc[df["HR"] < 45, "New_HR"] = "Low"
    df.loc[(df["HR"] >= 45) & (df["HR"] <= 100), "New_HR"] = "Normal"
    df.loc[df["HR"] > 100, "New_HR"] = "High"

    df.loc[df["RR"] < 12, "New_RR"] = "Low"
    df.loc[(df["RR"] >= 12) & (df["RR"] <= 25), "New_RR"] = "Normal"
    df.loc[df["RR"] > 25, "New_RR"] = "High"

    df.loc[df["BT"] < 36.4, "New_BT"] = "Low"
    df.loc[(df["BT"] >= 36.4) & (df["BT"] <= 37.6), "New_BT"] = "Normal"
    df.loc[df["BT"] > 37.6, "New_BT"] = "High"

    df.loc[df["NRS_pain"] < 3, "New_NRS_pain"] = "Low Pain"
    df.loc[(df["NRS_pain"] >= 3) & (df["NRS_pain"] <= 7), "New_NRS_pain"] = "Pain"
    df.loc[df["NRS_pain"] > 7, "New_NRS_pain"] = "High Pain"

    df.loc[df["KTAS duration_min"] < 10, "New_KTAS_duration_min"] = "Immediate"
    df.loc[
        (df["KTAS duration_min"] >= 10) & (df["KTAS duration_min"] <= 60),
        "New_KTAS_duration_min",
    ] = "Very Urgent"
    df.loc[
        (df["KTAS duration_min"] >= 61) & (df["KTAS duration_min"] <= 120),
        "New_KTAS_duration_min",
    ] = "Urgent"
    df.loc[
        (df["KTAS duration_min"] >= 121) & (df["KTAS duration_min"] <= 240),
        "New_KTAS_duration_min",
    ] = "Standart"
    df.loc[df["KTAS duration_min"] > 240, "New_KTAS_duration_min"] = "Non-Urgent"

    df.loc[df["Length of stay_min"] < 10, "New_Length_of_stay_min"] = "Immediate"
    df.loc[
        (df["Length of stay_min"] >= 10) & (df["Length of stay_min"] <= 60),
        "New_Length_of_stay_min",
    ] = "Very Urgent"
    df.loc[
        (df["Length of stay_min"] >= 61) & (df["Length of stay_min"] <= 120),
        "New_Length_of_stay_min",
    ] = "Urgent"
    df.loc[
        (df["Length of stay_min"] >= 121) & (df["Length of stay_min"] <= 240),
        "New_Length_of_stay_min",
    ] = "Standart"
    df.loc[df["Length of stay_min"] > 240, "New_Length_of_stay_min"] = "Non-Urgent"

    return df


def robust_scaler(variable: pd.Series) -> pd.Series:
    var_median = variable.median()
    quartile1 = variable.quantile(0.05)
    quartile3 = variable.quantile(0.95)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return np.round(z, 3)
    z = (variable - var_median) / interquantile_range
    return np.round(z, 3)


def one_hot_encoder(
    dataframe: pd.DataFrame, categorical_columns: list, nan_as_category: bool = False
) -> tuple[pd.DataFrame, list]:
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(
        dataframe, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True
    )
    new_columns = [col for col in dataframe.columns if col not in original_columns]
    return dataframe, new_columns


def build_new_df(df: pd.DataFrame) -> pd.DataFrame:
    """Same column selection as notebook `new_df` (includes mistriage)."""
    cols = [
        "Group",
        "Sex",
        "Patients number per hour",
        "Arrival mode",
        "Injury",
        "Mental",
        "Pain",
        "Saturation",
        "KTAS_RN",
        "Disposition",
        "KTAS_expert",
        "Length of stay_min",
        "mistriage",
        "New_Age",
        "New_SBP",
        "New_DBP",
        "New_HR",
        "New_RR",
        "New_BT",
        "New_NRS_pain",
        "New_KTAS_duration_min",
        "New_Length_of_stay_min",
    ]
    return df[cols].copy()


def preprocess_notebook_style(new_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """like_num + robust_scaler + get_dummies (notebook 5.1)."""
    df = new_df.copy()
    like_num = [
        col
        for col in df.columns
        if df[col].dtype != object and len(df[col].value_counts()) > 10
    ]
    cols_need_scale = [col for col in df.columns if col not in like_num and col != "mistriage"]

    for col in like_num:
        df[col] = robust_scaler(df[col])

    df, _ = one_hot_encoder(df, cols_need_scale)
    feature_cols = [c for c in df.columns if c != "mistriage"]
    return df, feature_cols, like_num


def prepare_from_raw(path: Path | None = None) -> pd.DataFrame:
    """Full dataframe from CSV through notebook flow (includes target column)."""
    df = load_raw(path)
    df = _replace_bad_nrs(df)
    df = _fill_missing_notebook(df)
    df = _numeric_to_categories(df)
    df = _feature_engineering(df)
    return build_new_df(df)


def train_and_save(
    out_path: Path | None = None,
    random_state: int = 357,
    test_size: float = 0.2,
) -> dict:
    """
    Same train/test split as notebook; RF with strong defaults (no grid search).
    Preprocessing on full data (notebook-aligned).
    """
    out_path = out_path or Path(__file__).resolve().parent.parent / "models" / "mistriage_kaggle_rf.joblib"

    prepared = prepare_from_raw()
    new_df = prepared
    processed, feature_cols, like_num = preprocess_notebook_style(new_df)

    X = processed.drop("mistriage", axis=1)
    y = np.ravel(processed[["mistriage"]])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        max_features=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    train_acc = float(clf.score(X_train, y_train))
    test_acc = float(clf.score(X_test, y_test))

    bundle = {
        "model": clf,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "like_num_columns": like_num,
        "classes": le.classes_.tolist(),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "random_state": random_state,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return {**bundle, "path": str(out_path)}


def build_raw_input_row(
    *,
    group: int,
    sex: int,
    age: int,
    patients_per_hour: int,
    arrival_mode: int,
    injury: int,
    chief_complain: str,
    mental: int,
    pain: int,
    nrs_pain: float | None,
    sbp: float,
    dbp: float,
    hr: float,
    rr: float,
    bt: float,
    saturation: float,
    ktas_rn: int,
    diagnosis_ed: str,
    disposition: int,
    ktas_expert: int,
    error_group: int,
    length_of_stay_min: float,
    ktas_duration_min: float,
) -> pd.DataFrame:
    """One row in notebook raw schema (same columns as `load_raw`)."""
    row = {
        "Group": group,
        "Sex": sex,
        "Age": age,
        "Patients number per hour": patients_per_hour,
        "Arrival mode": arrival_mode,
        "Injury": injury,
        "Chief_complain": chief_complain,
        "Mental": mental,
        "Pain": pain,
        "NRS_pain": float(nrs_pain) if nrs_pain is not None else np.nan,
        "SBP": sbp,
        "DBP": dbp,
        "HR": hr,
        "RR": rr,
        "BT": bt,
        "Saturation": saturation,
        "KTAS_RN": ktas_rn,
        "Diagnosis in ED": diagnosis_ed,
        "Disposition": disposition,
        "KTAS_expert": ktas_expert,
        "Error_group": error_group,
        "Length of stay_min": length_of_stay_min,
        "KTAS duration_min": ktas_duration_min,
        "mistriage": 0,
    }
    return pd.DataFrame([row])


def predict_mistriage(bundle: dict, X_aligned: pd.DataFrame) -> tuple[str, np.ndarray, list[str]]:
    """Single row aligned with training columns. `proba` order matches `class_names`."""
    model: RandomForestClassifier = bundle["model"]
    le: LabelEncoder = bundle["label_encoder"]
    pred_enc = int(model.predict(X_aligned)[0])
    label = le.inverse_transform([pred_enc])[0]
    proba = model.predict_proba(X_aligned)[0]
    class_names = [str(le.inverse_transform([int(c)])[0]) for c in model.classes_]
    return str(label), proba, class_names


def build_X_for_prediction(
    bundle: dict,
    new_df_single: pd.DataFrame,
    full_reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Single sample: notebook preprocessing on full reference + new row;
    returns feature vector of last row (mistriage column dropped).
    """
    full_reference_df = full_reference_df if full_reference_df is not None else prepare_from_raw()
    combined_with_y = new_df_single.copy()
    combined_with_y["mistriage"] = MISTRIAGE_LABELS[0]
    full_with_y = pd.concat([full_reference_df, combined_with_y], ignore_index=True)
    processed_full, _, _ = preprocess_notebook_style(full_with_y)
    X_row = processed_full.iloc[[-1]].drop(columns=["mistriage"])
    X_row = X_row.reindex(columns=bundle["feature_columns"], fill_value=0)
    return X_row


def raw_row_to_new_df_single(raw_row: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one raw CSV-schema row into notebook `new_df` format.
    Concatenated with full data for missing-value imputation.
    """
    full = load_raw()
    combined = pd.concat([full, raw_row], ignore_index=True)
    combined = _replace_bad_nrs(combined)
    combined = _fill_missing_notebook(combined)
    combined = _numeric_to_categories(combined)
    combined = _feature_engineering(combined)
    return build_new_df(combined).iloc[[-1]]


def load_mistriage_bundle(path: Path | str | None = None) -> dict:
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "models" / "mistriage_kaggle_rf.joblib"
    return joblib.load(path)


def predict_mistriage_from_raw_row(bundle: dict, raw_row: pd.DataFrame) -> tuple[str, np.ndarray, list[str]]:
    new_df_single = raw_row_to_new_df_single(raw_row)
    X = build_X_for_prediction(bundle, new_df_single)
    return predict_mistriage(bundle, X)


if __name__ == "__main__":
    out = train_and_save()
    print("Saved:", out["path"])
    print("Train accuracy:", round(out["train_accuracy"], 4))
    print("Test accuracy:", round(out["test_accuracy"], 4))
