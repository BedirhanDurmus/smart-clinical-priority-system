"""Data visualization charts (shared by app.py and optional multipage entry)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from src.load_data import DATA_PATH
from src.viz_charts import (
    fig_age_distribution,
    fig_ktas_side_by_side,
    fig_length_of_stay,
    fig_mistriage_bar,
    fig_patients_per_hour,
    fig_rn_expert_heatmap,
    fig_sex_group,
    fig_summary_metrics,
    fig_vitals_by_ktas,
    prepare_viz_df,
)


@st.cache_data(show_spinner=False)
def _load_viz_df():
    return prepare_viz_df()


def render_data_visualization() -> None:
    """Render the full training-data chart view."""
    st.title("📊 Project data — visualization")
    st.caption(
        "These charts summarize your **training/research dataset** (`data/data.csv`). "
        "They provide context on triage patterns, distributions, and model inputs."
    )

    if not DATA_PATH.exists():
        st.error(f"Data file not found: `{DATA_PATH}`")
        return

    df = _load_viz_df()

    st.subheader("Summary statistics")
    st.markdown(
        "**What is it?** At a glance: number of cases, overall missingness, and nurse–physician "
        "exact KTAS agreement — useful for data quality and reporting."
    )
    fig0 = fig_summary_metrics(df)
    st.pyplot(fig0)
    plt.close(fig0)

    st.markdown("---")

    st.subheader("1. Nurse and physician KTAS distributions")
    st.markdown(
        "**What?** KTAS 1–5 (1 = most urgent). **Why it matters:** Shows which urgency levels dominate; "
        "helps spot class imbalance or skew in what the model learned from."
    )
    fig1 = fig_ktas_side_by_side(df)
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")

    st.subheader("2. Triage agreement (mistriage)")
    st.markdown(
        "**What?** The `mistriage` column encodes whether nurse KTAS is **normal**, **over-**, or "
        "**under-** triage relative to physician KTAS. **Why it matters:** Error-type rates for training and quality review."
    )
    fig2 = fig_mistriage_bar(df)
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")

    st.subheader("3. Nurse × physician KTAS cross-tabulation")
    st.markdown(
        "**What?** Rows = nurse KTAS, columns = physician KTAS, cells = counts. **Why it matters:** "
        "Shows how often each mismatch pattern occurs (e.g. nurse 3 vs physician 4)."
    )
    fig3 = fig_rn_expert_heatmap(df)
    st.pyplot(fig3)
    plt.close(fig3)

    st.markdown("---")

    st.subheader("4. Age distribution")
    st.markdown(
        "**What?** Histogram with median line. **Why it matters:** Age profile of ED visits; baseline demography for the model."
    )
    fig4 = fig_age_distribution(df)
    st.pyplot(fig4)
    plt.close(fig4)

    st.markdown("---")

    st.subheader("5. Sex and hospital group")
    st.markdown(
        "**What?** Sex and local vs regional ED counts. **Why it matters:** Sample composition and fair comparison across sites."
    )
    fig5 = fig_sex_group(df)
    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown("---")

    st.subheader("6. Length of stay in the ED")
    st.markdown(
        "**What?** Histogram of `log(1 + Length of stay_min)` to soften long tails. **Why it matters:** "
        "Flow and workload; extreme stays inform triage and bed management discussions."
    )
    fig6 = fig_length_of_stay(df)
    st.pyplot(fig6)
    plt.close(fig6)

    st.markdown("---")

    st.subheader("7. Hourly patient load")
    st.markdown(
        "**What?** Distribution of patients per arrival hour. **Why it matters:** Busy hours may correlate with triage load and waits."
    )
    fig7 = fig_patients_per_hour(df)
    st.pyplot(fig7)
    plt.close(fig7)

    st.markdown("---")

    st.subheader("8. Vitals vs KTAS")
    st.markdown(
        "**What?** Boxplots of heart rate and systolic BP by nurse KTAS. **Why it matters:** "
        "Shows how vitals separate across triage levels alongside your model features."
    )
    fig8 = fig_vitals_by_ktas(df)
    st.pyplot(fig8)
    plt.close(fig8)

    st.info(
        "This view is **exploratory** on the CSV only. For clinical decisions, use the main app output "
        "and qualified physician judgment."
    )
