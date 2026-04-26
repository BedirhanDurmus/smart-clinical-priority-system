"""Academic presentation page for the ED triage AI project."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_data import load_raw
from src.predict import load_model, predict_single


st.set_page_config(page_title="Presentation", page_icon="🧭", layout="wide")


def _safe_int(v, default=0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v, default=np.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _predict_row(model, row: pd.Series) -> int:
    res = predict_single(
        model,
        age=_safe_int(row.get("Age"), 45),
        sex=_safe_int(row.get("Sex"), 1),
        group=_safe_int(row.get("Group"), 1),
        arrival_mode=_safe_int(row.get("Arrival mode"), 3),
        injury=_safe_int(row.get("Injury"), 1),
        chief_complaint=str(row.get("Chief_complain", "unknown")),
        mental=_safe_int(row.get("Mental"), 1),
        pain=_safe_int(row.get("Pain"), 0),
        nrs_pain=_safe_float(row.get("NRS_pain"), np.nan),
        sbp=_safe_float(row.get("SBP"), np.nan),
        dbp=_safe_float(row.get("DBP"), np.nan),
        hr=_safe_float(row.get("HR"), np.nan),
        rr=_safe_float(row.get("RR"), np.nan),
        bt=_safe_float(row.get("BT"), np.nan),
        saturation=_safe_float(row.get("Saturation"), np.nan),
        patients_per_hour=_safe_int(row.get("Patients number per hour"), 5),
    )
    return int(res["predicted_ktas"])


@st.cache_data(show_spinner=False)
def _analysis_cache():
    df = load_raw()
    model_path = ROOT / "models" / "best_model.joblib"
    if not model_path.exists():
        return df, None, None, None

    model = load_model(model_path)
    preds = []
    for _, r in df.iterrows():
        preds.append(_predict_row(model, r))

    work = df.copy()
    work["ai_ktas"] = preds
    work["KTAS_RN_num"] = pd.to_numeric(work["KTAS_RN"], errors="coerce")
    work["KTAS_expert_num"] = pd.to_numeric(work["KTAS_expert"], errors="coerce")
    work = work.dropna(subset=["KTAS_RN_num", "KTAS_expert_num"]).copy()
    work["KTAS_RN_num"] = work["KTAS_RN_num"].astype(int)
    work["KTAS_expert_num"] = work["KTAS_expert_num"].astype(int)

    # Potentially "corrected" cases: nurse != expert AND AI == expert
    corrected = work[(work["KTAS_RN_num"] != work["KTAS_expert_num"]) & (work["ai_ktas"] == work["KTAS_expert_num"])]
    total_nurse_errors = int((work["KTAS_RN_num"] != work["KTAS_expert_num"]).sum())
    corrected_count = int(len(corrected))
    corrected_ratio = (corrected_count / total_nurse_errors) if total_nurse_errors else 0.0

    # Pick one demonstrative case
    sample = corrected.iloc[0] if corrected_count else work.iloc[0]

    metrics = {
        "n": int(len(work)),
        "nurse_agreement": float((work["KTAS_RN_num"] == work["KTAS_expert_num"]).mean()),
        "ai_agreement": float((work["ai_ktas"] == work["KTAS_expert_num"]).mean()),
        "nurse_errors": total_nurse_errors,
        "potentially_corrected": corrected_count,
        "potentially_corrected_ratio": corrected_ratio,
    }
    return work, sample, metrics, model


st.title("🧭 Project Presentation — ED Triage AI")
st.caption("Methodology, model development, and operational implications")

st.markdown(
    """
This page consolidates the full analytical workflow:
- dataset loading and preprocessing choices,
- exploratory analysis and feature engineering strategy,
- model training and validation protocol,
- deployment behavior in the Streamlit interface,
- retrospective estimate of potential triage-error mitigation.
"""
)

st.markdown("---")
st.header("1) Problem Definition and Operational Logic")
st.markdown(
    """
**Clinical-operational problem.** Triage misclassification in emergency departments has two principal error modes:
- **Overtriage:** urgency inflation, typically associated with avoidable resource consumption.
- **Undertriage:** urgency deflation, associated with elevated patient-safety risk.

**Reference assumption used in this project.**  
For quality-analysis purposes, **expert KTAS is treated as the reference label**.  
Accordingly, nurse and AI outputs are evaluated by their agreement with expert assignment.

**Operational logic.**  
Given first-contact clinical variables, the model outputs a **reference KTAS estimate**.  
Nurse and AI labels are then compared against expert labeling to quantify potential error interception.
"""
)

st.markdown("---")
st.header("2) Notebook Pipeline (End-to-End)")
st.markdown(
    """
### A. Data preparation
1. Loaded `data/data.csv` with delimiter and decimal normalization.  
2. Cleaned missing/invalid tokens (`??`, malformed `NRS_pain`).  
3. Standardized categorical fields (Sex, Injury, Mental status, Arrival mode, etc.).

### B. Exploratory analysis
1. KTAS distributions (nurse vs expert),  
2. mistriage distributions (normal/over/under),  
3. age, vitals, length-of-stay, and workload distributions.

### C. Feature engineering
1. Normalized chief-complaint text and mapped it to categories.  
2. Constructed model inputs from vitals, demographics, and operational context.  
3. Applied encoding/scaling in a reproducible preprocessing pipeline.

### D. Modeling
1. Trained a multiclass classifier for KTAS estimation,  
2. evaluated with hold-out testing and cross-validation,  
3. persisted the selected artifact as `models/best_model.joblib`.

### E. Productization
1. Deployed an interactive Streamlit inference interface,  
2. added a dedicated visualization page for dataset diagnostics,  
3. added this presentation page for technical and stakeholder communication.
"""
)

st.markdown("---")
st.header("3) Feature Rationale")
feature_df = pd.DataFrame(
    [
        ("Age, Sex", "Baseline risk profile and cohort heterogeneity"),
        ("Arrival mode", "Proxy indicator of acute severity at presentation"),
        ("Mental status", "Strong urgency-related clinical signal"),
        ("Pain + NRS", "Quantified symptom intensity"),
        ("SBP/DBP/HR/RR/BT/SpO2", "Core physiological state descriptors"),
        ("Patients per hour", "Operational congestion context"),
        ("Chief complaint", "Clinical context signal derived from intake text"),
        ("Nurse vs Expert KTAS", "Endpoint for discordance and quality analysis"),
    ],
    columns=["Variable block", "Rationale"],
)
st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.header("4) Application Workflow")
st.markdown(
    """
1. Populate the patient intake form in the **App** page.  
2. Execute **Run analysis** to obtain the model-estimated KTAS reference.  
3. Enter nurse and physician KTAS levels to evaluate directional concordance versus model output.  
4. Use results as decision support; final authority remains with clinical staff.
"""
)

st.markdown("---")
st.header("5) Real-Case Illustration: Nurse vs AI")
work, sample, metrics, _ = _analysis_cache()

if metrics is None:
    st.warning(
        "Model artifact not found (`models/best_model.joblib`). "
        "Train the model to enable this section."
    )
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyzed records", f"{metrics['n']}")
    c2.metric("Nurse–Expert agreement", f"{metrics['nurse_agreement']*100:.1f}%")
    c3.metric("AI–Expert agreement", f"{metrics['ai_agreement']*100:.1f}%")

    agreement_delta = (metrics["ai_agreement"] - metrics["nurse_agreement"]) * 100
    if agreement_delta >= 0:
        st.success(f"AI–Expert agreement is {agreement_delta:.1f} points above Nurse–Expert agreement.")
    else:
        st.info(
            f"AI–Expert agreement is {abs(agreement_delta):.1f} points below Nurse–Expert agreement "
            "in this artifact; however, AI may still capture a substantial subset of discordant nurse cases."
        )

    st.markdown(
        f"""
**Retrospective impact estimate (dataset-internal):**
- Nurse–expert discordant cases: **{metrics['nurse_errors']}**
- Discordant cases where AI matches expert: **{metrics['potentially_corrected']}**
- Potential capture ratio: **{metrics['potentially_corrected_ratio']*100:.1f}%**
"""
    )

    st.subheader("Why use AI over unaided nurse-only triage?")
    st.markdown(
        f"""
Under the expert-as-reference assumption, the primary value proposition is **error interception**:
- AI acts as a real-time second reader and can flag a meaningful fraction of nurse–expert discordances.
- In this dataset, AI aligns with expert in **{metrics['potentially_corrected']} / {metrics['nurse_errors']}**
  discordant nurse cases (**{metrics['potentially_corrected_ratio']*100:.1f}% potential capture**).
- Therefore, the proposed deployment model is **AI-assisted triage quality control**, not nurse replacement.
"""
    )

    st.subheader("Representative case profile")
    if sample is None or getattr(sample, "empty", False):
        st.info("No representative case could be generated from the current data subset.")
    else:
        ex = {
            "Chief complaint": str(sample.get("Chief_complain", "N/A")),
            "Age": _safe_int(sample.get("Age"), -1),
            "Mental status code": _safe_int(sample.get("Mental"), -1),
            "Pain code": _safe_int(sample.get("Pain"), -1),
            "SBP/DBP": f"{sample.get('SBP', 'N/A')}/{sample.get('DBP', 'N/A')}",
            "HR": sample.get("HR", "N/A"),
            "RR": sample.get("RR", "N/A"),
            "SpO2": sample.get("Saturation", "N/A"),
            "Nurse KTAS": _safe_int(sample.get("KTAS_RN_num"), -1),
            "Expert KTAS": _safe_int(sample.get("KTAS_expert_num"), -1),
            "AI KTAS": _safe_int(sample.get("ai_ktas"), -1),
        }
        st.json(ex)

        nurse = _safe_int(sample.get("KTAS_RN_num"), -1)
        expert = _safe_int(sample.get("KTAS_expert_num"), -1)
        ai = _safe_int(sample.get("ai_ktas"), -1)

        st.markdown("#### Static prediction-style result panel (non-interactive)")
        st.markdown(
            f"""
<div style="border:1px solid #e2e8f0;border-radius:12px;padding:14px;background:#f8fafc;">
  <div style="font-weight:700;margin-bottom:8px;">Case outcome snapshot</div>
  <div>🏥 <strong>Nurse KTAS:</strong> {nurse}</div>
  <div>🧠 <strong>AI KTAS:</strong> {ai}</div>
  <div>🩺 <strong>Expert KTAS (reference):</strong> {expert}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if nurse != expert and ai == expert:
            st.success(
                "In this case, nurse assignment is discordant with expert labeling, "
                "whereas AI aligns with expert KTAS. This illustrates potential early error interception."
            )
        elif nurse == expert:
            st.info("In this case, nurse assignment is already concordant with expert KTAS.")
        else:
            st.warning(
                "In this case, AI is also discordant with expert KTAS. "
                "This is consistent with expected model limitations."
            )

st.markdown("---")
st.header("6) Limitations and Appropriate Use")
st.markdown(
    """
- This system is a **decision-support** layer, not an autonomous decision maker.  
- Performance is distribution-dependent; out-of-distribution behavior must be monitored.  
- Clinical examination and physician judgment remain the definitive standard.  
- Continuous monitoring, periodic recalibration, and governance metrics are required.
"""
)

