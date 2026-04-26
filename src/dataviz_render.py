"""Data visualization charts (shared by app.py and optional multipage entry)."""

from __future__ import annotations

import inspect
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


def _render_code_card(
    title: str,
    explanation: str,
    call_code: str,
    fn_obj,
) -> None:
    """Show chart call + full function body under each chart."""
    st.markdown(
        f"""
<div style="
    margin-top:8px;
    margin-bottom:8px;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:10px 12px;
    background:#f8fafc;
">
  <div style="font-weight:700; margin-bottom:4px;">💻 {title}</div>
  <div style="font-size:0.92rem; color:#475569; line-height:1.45;">{explanation}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    try:
        fn_source = inspect.getsource(fn_obj)
    except Exception:
        fn_source = "# Function source could not be loaded in this environment."

    full_code = (
        "# Chart call in this page\n"
        f"{call_code}\n\n"
        "# Function internals (from src/viz_charts.py)\n"
        f"{fn_source}"
    )
    show_code = st.checkbox(
        "☑ Show/Hide code",
        value=False,
        key=f"code_toggle_{title}",
    )
    if show_code:
        st.code(full_code, language="python")


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
    _render_code_card(
        title="Summary statistics grafiği kodu",
        explanation="Bu blok özet metriği (vaka sayısı, eksik veri oranı, hemşire-doktor KTAS uyumu) "
        "tek bir figürde üretir ve Streamlit'e basar.",
        call_code="""fig0 = fig_summary_metrics(df)
st.pyplot(fig0)
plt.close(fig0)""",
        fn_obj=fig_summary_metrics,
    )

    st.markdown("---")

    st.subheader("1. Nurse and physician KTAS distributions")
    st.markdown(
        "**What?** KTAS 1–5 (1 = most urgent). **Why it matters:** Shows which urgency levels dominate; "
        "helps spot class imbalance or skew in what the model learned from."
    )
    fig1 = fig_ktas_side_by_side(df)
    st.pyplot(fig1)
    plt.close(fig1)
    _render_code_card(
        title="KTAS dağılımı (hemşire vs doktor) kodu",
        explanation="Hemşire ve doktor KTAS seviyelerini yan yana gösterir. "
        "Sınıf dengesizliği ve dağılım farklarını hızlı görürsünüz.",
        call_code="""fig1 = fig_ktas_side_by_side(df)
st.pyplot(fig1)
plt.close(fig1)""",
        fn_obj=fig_ktas_side_by_side,
    )

    st.markdown("---")

    st.subheader("2. Triage agreement (mistriage)")
    st.markdown(
        "**What?** The `mistriage` column encodes whether nurse KTAS is **normal**, **over-**, or "
        "**under-** triage relative to physician KTAS. **Why it matters:** Error-type rates for training and quality review."
    )
    fig2 = fig_mistriage_bar(df)
    st.pyplot(fig2)
    plt.close(fig2)
    _render_code_card(
        title="Mistriage bar grafiği kodu",
        explanation="Normal / over / under triage sınıflarının frekanslarını üretir; "
        "hata tipi profilini görmenizi sağlar.",
        call_code="""fig2 = fig_mistriage_bar(df)
st.pyplot(fig2)
plt.close(fig2)""",
        fn_obj=fig_mistriage_bar,
    )

    st.markdown("---")

    st.subheader("3. Nurse × physician KTAS cross-tabulation")
    st.markdown(
        "**What?** Rows = nurse KTAS, columns = physician KTAS, cells = counts. **Why it matters:** "
        "Shows how often each mismatch pattern occurs (e.g. nurse 3 vs physician 4)."
    )
    fig3 = fig_rn_expert_heatmap(df)
    st.pyplot(fig3)
    plt.close(fig3)
    _render_code_card(
        title="Hemşire × doktor heatmap kodu",
        explanation="Satırda hemşire, sütunda doktor KTAS olacak şekilde kesişim tablosunu ısı haritası olarak çizer.",
        call_code="""fig3 = fig_rn_expert_heatmap(df)
st.pyplot(fig3)
plt.close(fig3)""",
        fn_obj=fig_rn_expert_heatmap,
    )

    st.markdown("---")

    st.subheader("4. Age distribution")
    st.markdown(
        "**What?** Histogram with median line. **Why it matters:** Age profile of ED visits; baseline demography for the model."
    )
    fig4 = fig_age_distribution(df)
    st.pyplot(fig4)
    plt.close(fig4)
    _render_code_card(
        title="Yaş dağılımı kodu",
        explanation="Yaş histogramını ve medyan referansını çizer; örneklemin demografik yapısını okumanızı sağlar.",
        call_code="""fig4 = fig_age_distribution(df)
st.pyplot(fig4)
plt.close(fig4)""",
        fn_obj=fig_age_distribution,
    )

    st.markdown("---")

    st.subheader("5. Sex and hospital group")
    st.markdown(
        "**What?** Sex and local vs regional ED counts. **Why it matters:** Sample composition and fair comparison across sites."
    )
    fig5 = fig_sex_group(df)
    st.pyplot(fig5)
    plt.close(fig5)
    _render_code_card(
        title="Cinsiyet / hastane grubu kodu",
        explanation="Cinsiyet ve hastane tipi (local/regional ED) dağılımlarını karşılaştırmalı verir.",
        call_code="""fig5 = fig_sex_group(df)
st.pyplot(fig5)
plt.close(fig5)""",
        fn_obj=fig_sex_group,
    )

    st.markdown("---")

    st.subheader("6. Length of stay in the ED")
    st.markdown(
        "**What?** Histogram of `log(1 + Length of stay_min)` to soften long tails. **Why it matters:** "
        "Flow and workload; extreme stays inform triage and bed management discussions."
    )
    fig6 = fig_length_of_stay(df)
    st.pyplot(fig6)
    plt.close(fig6)
    _render_code_card(
        title="Acilde kalış süresi dağılımı kodu",
        explanation="Uzun kuyruk etkisini azaltmak için log-dönüşümlü kalış süresi histogramı üretir.",
        call_code="""fig6 = fig_length_of_stay(df)
st.pyplot(fig6)
plt.close(fig6)""",
        fn_obj=fig_length_of_stay,
    )

    st.markdown("---")

    st.subheader("7. Hourly patient load")
    st.markdown(
        "**What?** Distribution of patients per arrival hour. **Why it matters:** Busy hours may correlate with triage load and waits."
    )
    fig7 = fig_patients_per_hour(df)
    st.pyplot(fig7)
    plt.close(fig7)
    _render_code_card(
        title="Saatlik hasta yükü kodu",
        explanation="Saat başına hasta sayısının dağılımını gösterir; yoğun saatleri yorumlamaya yardımcı olur.",
        call_code="""fig7 = fig_patients_per_hour(df)
st.pyplot(fig7)
plt.close(fig7)""",
        fn_obj=fig_patients_per_hour,
    )

    st.markdown("---")

    st.subheader("8. Vitals vs KTAS")
    st.markdown(
        "**What?** Boxplots of heart rate and systolic BP by nurse KTAS. **Why it matters:** "
        "Shows how vitals separate across triage levels alongside your model features."
    )
    fig8 = fig_vitals_by_ktas(df)
    st.pyplot(fig8)
    plt.close(fig8)
    _render_code_card(
        title="Vital bulgular vs KTAS kodu",
        explanation="Kalp hızı ve sistolik tansiyonun KTAS seviyelerine göre kutu grafikleriyle nasıl ayrıştığını gösterir.",
        call_code="""fig8 = fig_vitals_by_ktas(df)
st.pyplot(fig8)
plt.close(fig8)""",
        fn_obj=fig_vitals_by_ktas,
    )

    st.info(
        "This view is **exploratory** on the CSV only. For clinical decisions, use the main app output "
        "and qualified physician judgment."
    )
