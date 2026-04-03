"""Run the full pipeline: load → preprocess → improved training → save."""

import numpy as np
import pandas as pd

from src.load_data import load_raw, quick_summary
from src.chief_complaint import add_complaint_features
from src.preprocess import (
    prepare_dataframe,
    get_feature_columns,
    TARGET,
)
from src.train import run_improved_training
from src.predict import load_model, predict_single


def main():
    print("=" * 70)
    print("STEP 1: Loading data")
    print("=" * 70)
    df_raw = load_raw()
    summary = quick_summary(df_raw)
    print(f"Shape: {df_raw.shape}")
    print(f"Missing cols with >0: {sum(1 for v in summary['missing_per_col'].values() if v > 0)}")

    sentinels = ["??", "#BO\ufffd!", "#BO\xde!"]
    for col in df_raw.columns:
        vals = df_raw[col].astype(str).str.strip()
        for s in sentinels:
            cnt = (vals == s).sum()
            if cnt > 0:
                print(f"  Sentinel in {col}: {cnt}")

    print("\n" + "=" * 70)
    print("STEP 2: Chief Complaint analysis")
    print("=" * 70)
    df_cc = add_complaint_features(df_raw)
    cat_dist = df_cc["cc_category"].value_counts()
    print("Category distribution:")
    print(cat_dist.to_string())
    print(f"\nUnique normalised complaints: {df_cc['cc_normalised'].nunique()}")
    print(f"Unknown complaints: {(df_cc['cc_normalised'] == 'unknown').sum()}")

    print("\n" + "=" * 70)
    print("STEP 3: Preprocessing + improved model (TF-IDF, hold-out, tuning)")
    print("=" * 70)
    df = prepare_dataframe(df_raw)
    print(f"After cleaning: {df.shape}")
    print(f"Target distribution:\n{df[TARGET].value_counts().sort_index()}")

    X_cols = get_feature_columns()
    print(f"Feature columns ({len(X_cols)}): {X_cols}")

    result = run_improved_training(df, test_size=0.2, search_n_iter=45)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Best CV macro-F1 (LGBM search): {result['best_cv_macro_f1']:.4f}")
    print(f"Deployed model: {result['chosen_model']}")
    print(
        f"Hold-out — F1-macro={result['chosen_holdout']['test_f1_macro']}, "
        f"gap={result['chosen_holdout']['gap_f1_macro']}"
    )
    print(f"Model path: {result['model_path']}")

    print("\nBaseline train vs test (same split):")
    for row in result["baseline_table"]:
        print(
            f"  {row['model']}: test F1-macro={row['test_f1_macro']}, "
            f"gap={row['gap_f1_macro']}"
        )

    print("\n" + "=" * 70)
    print("STEP 4: Quick prediction test")
    print("=" * 70)
    model = load_model()
    r1 = predict_single(
        model,
        age=55, sex=2, group=1, arrival_mode=3, injury=1,
        chief_complaint="chest pain",
        mental=1, pain=1, nrs_pain=5,
        sbp=140, dbp=90, hr=88, rr=20, bt=36.5, saturation=98,
    )
    print("Chest pain demo:", r1)

    print("\n" + "=" * 70)
    print("STEP 5: Secondary analysis — KTAS_RN vs KTAS_expert")
    print("=" * 70)
    df_sec = df_raw.copy()
    df_sec["KTAS_RN_num"] = pd.to_numeric(df_sec["KTAS_RN"], errors="coerce")
    df_sec["KTAS_expert_num"] = pd.to_numeric(df_sec["KTAS_expert"], errors="coerce")
    df_sec = df_sec.dropna(subset=["KTAS_RN_num", "KTAS_expert_num"])
    agreement = (df_sec["KTAS_RN_num"] == df_sec["KTAS_expert_num"]).mean()
    print(f"RN vs Expert agreement: {agreement:.2%}")

    df_sec["mistriage_clean"] = pd.to_numeric(df_sec["mistriage"].astype(str).str.strip(), errors="coerce")
    mis = df_sec["mistriage_clean"].dropna().astype(int).value_counts().sort_index()
    print(f"Mistriage distribution: {mis.to_dict()}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
