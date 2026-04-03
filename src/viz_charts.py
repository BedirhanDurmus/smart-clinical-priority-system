"""Matplotlib/seaborn figures for the Streamlit data visualization page."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.load_data import load_raw

MISTRIAGE_LABELS = {
    0: "Normal triage",
    1: "Overtriage",
    2: "Undertriage",
}

SEX_LABELS = {1: "Female", 2: "Male"}


def prepare_viz_df() -> pd.DataFrame:
    df = load_raw().copy()
    num_cols = [
        "KTAS_RN",
        "KTAS_expert",
        "mistriage",
        "Sex",
        "Group",
        "Age",
        "SBP",
        "DBP",
        "HR",
        "RR",
        "BT",
        "Saturation",
        "NRS_pain",
        "Patients number per hour",
        "Length of stay_min",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _style_axes():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def fig_summary_metrics(df: pd.DataFrame) -> plt.Figure:
    """Sample count, missing rate, nurse–physician exact agreement."""
    _style_axes()
    n = len(df)
    both = df[["KTAS_RN", "KTAS_expert"]].dropna()
    agree = (both["KTAS_RN"] == both["KTAS_expert"]).mean() * 100 if len(both) else 0
    miss = df.isnull().mean().mean() * 100

    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    texts = [
        f"Records: {n:,}",
        f"Mean missing cell rate: {miss:.1f}%",
        f"Nurse–physician same KTAS: {agree:.1f}%",
    ]
    ax.text(
        0.5,
        0.5,
        "   ·   ".join(texts),
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
        wrap=True,
    )
    fig.suptitle("Dataset summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig_ktas_side_by_side(df: pd.DataFrame) -> plt.Figure:
    """Nurse vs physician KTAS distributions (1–5)."""
    _style_axes()
    levels = [1, 2, 3, 4, 5]
    rn = df["KTAS_RN"].dropna().astype(int)
    ex = df["KTAS_expert"].dropna().astype(int)
    c_rn = rn.value_counts().reindex(levels, fill_value=0)
    c_ex = ex.value_counts().reindex(levels, fill_value=0)

    x = np.arange(len(levels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, c_rn.values, w, label="Nurse (KTAS_RN)", color="#2980b9", alpha=0.9)
    ax.bar(x + w / 2, c_ex.values, w, label="Physician (KTAS_expert)", color="#27ae60", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"KTAS {k}" for k in levels])
    ax.set_ylabel("Count")
    ax.set_xlabel("Urgency level (1 = most urgent)")
    ax.legend(loc="upper right")
    ax.set_title("Nurse and physician KTAS distributions")
    plt.tight_layout()
    return fig


def fig_mistriage_bar(df: pd.DataFrame) -> plt.Figure:
    """mistriage: 0 normal, 1 overtriage, 2 undertriage (nurse vs physician reference)."""
    _style_axes()
    m = df["mistriage"].dropna()
    m = m.astype(int)
    counts = m.value_counts().reindex([0, 1, 2], fill_value=0)
    labels = [MISTRIAGE_LABELS.get(i, str(i)) for i in [0, 1, 2]]
    colors = ["#2ecc71", "#e67e22", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Count")
    ax.set_title("Triage agreement (mistriage): nurse vs physician KTAS")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(v)}", ha="center", va="bottom", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def fig_rn_expert_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Nurse KTAS (rows) × physician KTAS (columns) confusion-style matrix."""
    _style_axes()
    sub = df[["KTAS_RN", "KTAS_expert"]].dropna()
    sub = sub.astype(int)
    sub = sub[(sub["KTAS_RN"].between(1, 5)) & (sub["KTAS_expert"].between(1, 5))]
    ct = pd.crosstab(sub["KTAS_RN"], sub["KTAS_expert"])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Physician KTAS (KTAS_expert)")
    ax.set_ylabel("Nurse KTAS (KTAS_RN)")
    ax.set_title("Nurse × physician KTAS cross-tabulation")
    plt.tight_layout()
    return fig


def fig_age_distribution(df: pd.DataFrame) -> plt.Figure:
    _style_axes()
    age = df["Age"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(age, bins=35, kde=True, color="#8e44ad", ax=ax, edgecolor="white")
    ax.axvline(age.median(), color="#c0392b", linestyle="--", linewidth=2, label=f"Median: {age.median():.0f}")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Age distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def fig_sex_group(df: pd.DataFrame) -> plt.Figure:
    _style_axes()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    sex = df["Sex"].dropna().astype(int)
    s_counts = sex.value_counts().reindex([1, 2], fill_value=0)
    ax1.bar(
        [SEX_LABELS[1], SEX_LABELS[2]],
        s_counts.values,
        color=["#9b59b6", "#34495e"],
    )
    ax1.set_ylabel("Count")
    ax1.set_title("Sex")

    grp = df["Group"].dropna().astype(int)
    g_counts = grp.value_counts().reindex([1, 2], fill_value=0)
    ax2.bar(
        ["Local\n(3rd)", "Regional\n(4th)"],
        g_counts.values,
        color=["#16a085", "#d35400"],
    )
    ax2.set_ylabel("Count")
    ax2.set_title("Hospital group")

    fig.suptitle("Demographics: sex and hospital type", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig_length_of_stay(df: pd.DataFrame) -> plt.Figure:
    """ED length of stay (minutes); log scale softens heavy tails."""
    _style_axes()
    los = df["Length of stay_min"].dropna()
    los = los[los >= 0]
    if len(los) == 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(9, 5))
    log_los = np.log1p(los)
    sns.histplot(log_los, bins=40, kde=True, color="#1abc9c", ax=ax)
    ax.set_xlabel("log(1 + minutes) — compresses long tails")
    ax.set_ylabel("Count")
    med = los.median()
    ax.set_title(f"Length of stay distribution (median ≈ {med:.0f} min)")
    plt.tight_layout()
    return fig


def fig_vitals_by_ktas(df: pd.DataFrame) -> plt.Figure:
    """Heart rate and SBP by nurse KTAS."""
    _style_axes()
    sub = df[["KTAS_RN", "HR", "SBP"]].dropna()
    sub = sub[sub["KTAS_RN"].between(1, 5)]
    sub["KTAS_RN"] = sub["KTAS_RN"].astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    order = [1, 2, 3, 4, 5]
    sns.boxplot(data=sub, x="KTAS_RN", y="HR", order=order, ax=ax1, palette="coolwarm")
    ax1.set_xlabel("Nurse KTAS")
    ax1.set_ylabel("Heart rate (bpm)")
    ax1.set_title("Heart rate × KTAS")

    sns.boxplot(data=sub, x="KTAS_RN", y="SBP", order=order, ax=ax2, palette="coolwarm")
    ax2.set_xlabel("Nurse KTAS")
    ax2.set_ylabel("Systolic BP (mmHg)")
    ax2.set_title("Systolic BP × KTAS")

    fig.suptitle("Vitals vs triage level", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def fig_patients_per_hour(df: pd.DataFrame) -> plt.Figure:
    """Hourly arrival load."""
    _style_axes()
    col = "Patients number per hour"
    if col not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Column missing", ha="center")
        return fig
    x = df[col].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(x, bins=20, kde=True, color="#e67e22", ax=ax)
    ax.set_xlabel("Patients per hour")
    ax.set_ylabel("Frequency")
    ax.set_title("ED arrival intensity (hourly)")
    plt.tight_layout()
    return fig
