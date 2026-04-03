"""
Emergency Department Triage Decision Support — Streamlit demo
1) Patient inputs → model KTAS prediction (reference: expected level per model)
2) Nurse and expert KTAS → each compared to the model reference (correct / wrong direction)
"""

import streamlit as st
from pathlib import Path

from src.predict import load_model, predict_single
from src.chief_complaint import CATEGORY_RULES, CHIEF_COMPLAINT_DROPDOWN

# ── Visual constants ─────────────────────────────────────────────

KTAS_COLORS = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71", 5: "#3498db"}

KTAS_EN = {
    1: "Resuscitation — Immediate life threat, treat now",
    2: "Emergent — Very urgent; limb/life risk possible",
    3: "Urgent — Serious but not immediately life-ending",
    4: "Less urgent — Can wait",
    5: "Non-urgent — Minor complaint",
}

# Result panel — neutral surfaces (light theme + readable in Streamlit dark)
SURFACE = "#f1f5f9"
SURFACE_ELEVATED = "#ffffff"
TEXT_PRIMARY = "#0f172a"
TEXT_MUTED = "#64748b"
BORDER_SUBTLE = "1px solid #e2e8f0"


def triage_verdict_style(assigned: int, model_ktas: int) -> dict:
    """
    KTAS: 1 = most urgent. Reference = model prediction.
    Match → green; overtriage → orange; undertriage → red.
    """
    if assigned == model_ktas:
        return {
            "kind": "match",
            "strip_title": "NORMAL TRIAGE",
            "strip_sub": "Correct ✓",
            "strip_bg": "#15803d",
            "strip_sub_bg": "#dcfce7",
            "strip_sub_color": "#166534",
            "card_bg": "#f0fdf4",
            "border": "#22c55e",
            "hint": "Assigned KTAS matches the model reference.",
        }
    if assigned < model_ktas:
        return {
            "kind": "over",
            "strip_title": "OVERTRIAGE",
            "strip_sub": "Incorrect",
            "strip_bg": "#c2410c",
            "strip_sub_bg": "#ffedd5",
            "strip_sub_color": "#9a3412",
            "card_bg": "#fff7ed",
            "border": "#fb923c",
            "hint": "Classified as more urgent than the model (lower KTAS number).",
        }
    return {
        "kind": "under",
        "strip_title": "UNDERTRIAGE",
        "strip_sub": "Incorrect",
        "strip_bg": "#b91c1c",
        "strip_sub_bg": "#fee2e2",
        "strip_sub_color": "#991b1b",
        "card_bg": "#fef2f2",
        "border": "#f87171",
        "hint": "Classified as less urgent than the model (higher KTAS number).",
    }


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_results_styles() -> str:
    """Typography and card shadows for the results section."""
    return """
<style>
  .tri-results-wrap {
    font-family: system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    letter-spacing: -0.01em;
  }
  .tri-card {
    background: """ + SURFACE_ELEVATED + """;
    border-radius: 16px;
    border: """ + BORDER_SUBTLE + """;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    overflow: hidden;
  }
  .tri-hero {
    position: relative;
    padding: 28px 24px 32px;
    text-align: center;
    color: #fff;
  }
  .tri-hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.92;
    margin-bottom: 8px;
  }
  .tri-hero-num {
    font-size: clamp(2.8rem, 8vw, 3.75rem);
    font-weight: 800;
    line-height: 1;
    margin: 0;
    text-shadow: 0 2px 20px rgba(0,0,0,0.15);
  }
  .tri-hero-desc {
    margin: 14px 0 0;
    font-size: 1.02rem;
    line-height: 1.45;
    font-weight: 500;
    opacity: 0.98;
    max-width: 28em;
    margin-left: auto;
    margin-right: auto;
  }
  .tri-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: """ + TEXT_MUTED + """;
    margin: 0 0 14px 0;
  }
  .tri-prose {
    color: """ + TEXT_PRIMARY + """;
    font-size: 0.94rem;
    line-height: 1.65;
  }
  .tri-prose ul { margin: 8px 0 0 0; padding-left: 1.15em; }
  .tri-prose li { margin-bottom: 6px; }
  .tri-ref-banner {
    background: linear-gradient(135deg, """ + SURFACE + """ 0%, """ + SURFACE_ELEVATED + """ 100%);
    border: """ + BORDER_SUBTLE + """;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  .tri-ref-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: """ + TEXT_MUTED + """;
    margin: 0 0 4px 0;
  }
  .tri-ref-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: """ + TEXT_PRIMARY + """;
    letter-spacing: -0.03em;
  }
  .tri-ref-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
    vertical-align: middle;
  }
  .tri-cc-v2 {
    border-radius: 14px;
    padding: 0 0 14px 0;
    margin-bottom: 16px;
    border: """ + BORDER_SUBTLE + """;
    box-shadow: 0 2px 14px rgba(15, 23, 42, 0.06);
    overflow: hidden;
  }
  .tri-cc-head {
    padding: 14px 16px 10px;
    font-size: 0.95rem;
    font-weight: 700;
    color: """ + TEXT_PRIMARY + """;
  }
  .tri-cc-vs {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    padding: 4px 16px 14px;
  }
  .tri-cc-vsblk {
    flex: 1;
    min-width: 120px;
    text-align: center;
    padding: 12px 10px;
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    border: 1px solid #e2e8f0;
  }
  .tri-cc-vsblk .tri-cc-mut {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: """ + TEXT_MUTED + """;
    margin-bottom: 6px;
  }
  .tri-cc-vsblk .tri-cc-num {
    font-size: 1.65rem;
    font-weight: 800;
    color: """ + TEXT_PRIMARY + """;
    letter-spacing: -0.03em;
  }
  .tri-cc-arrow {
    font-size: 1.4rem;
    color: """ + TEXT_MUTED + """;
    font-weight: 300;
    flex-shrink: 0;
  }
  .tri-cc-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    padding: 12px 16px;
    color: #fff;
  }
  .tri-cc-strip-title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.08em;
  }
  .tri-cc-subpill {
    font-size: 0.78rem;
    font-weight: 800;
    padding: 6px 12px;
    border-radius: 999px;
  }
  .tri-cc-hint {
    margin: 0;
    padding: 10px 16px 0;
    font-size: 0.86rem;
    color: """ + TEXT_MUTED + """;
    line-height: 1.45;
  }
  .tri-legend {
    background: """ + SURFACE + """;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 0.85rem;
    color: """ + TEXT_MUTED + """;
    line-height: 1.6;
    border: """ + BORDER_SUBTLE + """;
  }
  .tri-legend strong { color: """ + TEXT_PRIMARY + """; }
  .tri-bar-row { margin-bottom: 12px; }
  .tri-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
    color: """ + TEXT_MUTED + """;
    margin-bottom: 6px;
  }
  .tri-bar-track {
    height: 10px;
    background: """ + SURFACE + """;
    border-radius: 999px;
    overflow: hidden;
  }
  .tri-bar-fill { height: 100%; border-radius: 999px; transition: width 0.3s ease; }
  .tri-footnote {
    font-size: 0.88rem;
    color: """ + TEXT_MUTED + """;
    line-height: 1.55;
    padding: 14px 16px;
    background: """ + SURFACE + """;
    border-radius: 12px;
    border: """ + BORDER_SUBTLE + """;
  }
  /* Equal-height side-by-side results (replaces uneven Streamlit columns) */
  .tri-results-pair { width: 100%; margin: 0; }
  .tri-results-pair-inner {
    display: flex;
    flex-direction: row;
    align-items: stretch;
    gap: 1.25rem;
    flex-wrap: wrap;
  }
  .tri-results-pane {
    flex: 1 1 0;
    min-width: min(100%, 320px);
    display: flex;
    flex-direction: column;
  }
  .tri-results-pane > .tri-results-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 100%;
  }
  .tri-results-pane-left .tri-card {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 100%;
  }
  .tri-card-body {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
  }
  .tri-results-pane-right .tri-results-wrap {
    background: """ + SURFACE_ELEVATED + """;
    border-radius: 16px;
    border: """ + BORDER_SUBTLE + """;
    box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    padding: 18px 18px 20px;
    height: 100%;
    box-sizing: border-box;
  }
  @media (max-width: 900px) {
    .tri-results-pair-inner { flex-direction: column; }
  }
</style>
"""


def render_probability_bars(probabilities: dict[int, float]) -> str:
    """Horizontal bars for KTAS class probabilities."""
    if not probabilities:
        return ""
    rows = []
    for k in sorted(probabilities.keys()):
        p = float(probabilities[k])
        pct = min(100.0, max(0.0, p * 100))
        col = KTAS_COLORS.get(k, "#64748b")
        pk = _html_escape(str(k))
        rows.append(
            f'<div class="tri-bar-row">'
            f'<div class="tri-bar-label"><span>KTAS {pk}</span><span>{p:.0%}</span></div>'
            f'<div class="tri-bar-track"><div class="tri-bar-fill" style="width:{pct:.1f}%;background:{col};"></div></div>'
            f"</div>"
        )
    return (
        '<div style="padding: 4px 4px 0 4px;">'
        f'<p class="tri-section-title" style="margin-bottom:16px;">Probability distribution</p>'
        + "".join(rows)
        + "</div>"
    )


def render_compare_card(
    role: str,
    role_emoji: str,
    assigned: int,
    model_ktas: int,
    style: dict,
) -> str:
    """Nurse / expert: assigned vs expected + triage verdict strip."""
    sb = style["strip_bg"]
    card_bg = style["card_bg"]
    border = style["border"]
    sub_bg = style["strip_sub_bg"]
    sub_c = style["strip_sub_color"]
    title = _html_escape(style["strip_title"])
    sub = _html_escape(style["strip_sub"])
    hint = _html_escape(style["hint"])
    return f"""
<div class="tri-cc-v2" style="background:{card_bg}; border-left: 4px solid {border};">
  <div class="tri-cc-head">{role_emoji} {_html_escape(role)}</div>
  <div class="tri-cc-vs">
    <div class="tri-cc-vsblk">
      <div class="tri-cc-mut">Assigned</div>
      <div class="tri-cc-num">KTAS {assigned}</div>
    </div>
    <div class="tri-cc-arrow">→</div>
    <div class="tri-cc-vsblk">
      <div class="tri-cc-mut">Expected (model)</div>
      <div class="tri-cc-num">KTAS {model_ktas}</div>
    </div>
  </div>
  <div class="tri-cc-strip" style="background:{sb};">
    <span class="tri-cc-strip-title">{title}</span>
    <span class="tri-cc-subpill" style="background:{sub_bg};color:{sub_c};">{sub}</span>
  </div>
  <p class="tri-cc-hint">{hint}</p>
</div>
"""


def render_left_prediction_block(
    pred: int,
    desc: str,
    color: str,
    probabilities: dict[int, float] | None,
) -> str:
    """Left column: KTAS card + explanation + probability bars."""
    d = _html_escape(desc)
    hero_grad = f"linear-gradient(165deg, {color} 0%, {color} 52%, rgba(0,0,0,0.25) 100%)"
    bars = render_probability_bars(probabilities or {})
    return f"""
<div class="tri-results-wrap">
<div class="tri-card">
  <div class="tri-hero" style="background:{hero_grad};">
    <div class="tri-hero-badge">Predicted urgency level</div>
    <p class="tri-hero-num">KTAS {pred}</p>
    <p class="tri-hero-desc">{d}</p>
  </div>
  <div class="tri-card-body" style="padding: 22px 22px 26px;">
    <p class="tri-section-title">What does this mean?</p>
    <div class="tri-prose">
      <p style="margin:0 0 12px 0;">The model estimates which KTAS level (1–5) best matches the
      information you entered.</p>
      <ul>
        <li><strong>KTAS 1–2:</strong> Life threat; urgent intervention</li>
        <li><strong>KTAS 3:</strong> Serious; not immediately lethal</li>
        <li><strong>KTAS 4–5:</strong> Can wait or minor complaint</li>
      </ul>
    </div>
    {bars}
  </div>
</div>
</div>
"""


def render_right_evaluation_block(
    pred: int,
    pred_color: str,
    ktas_rn: int,
    ktas_expert: int,
    nurse_style: dict,
    expert_style: dict,
) -> str:
    """Right column: reference strip + nurse/expert cards + legend."""
    n_card = render_compare_card(
        "Nurse triage", "👩‍⚕️", ktas_rn, pred, nurse_style
    )
    e_card = render_compare_card(
        "Physician assessment", "🩺", ktas_expert, pred, expert_style
    )
    return f"""
<div class="tri-results-wrap">
  <p class="tri-section-title" style="margin-bottom:6px;">Check against model reference</p>
  <p style="margin:0 0 18px 0; font-size:0.95rem; color:{TEXT_MUTED}; line-height:1.5;">
    <strong style="color:{TEXT_PRIMARY};">KTAS {pred}</strong> is the model output; nurse and physician
    levels are compared to this reference one by one.
  </p>
  <div class="tri-ref-banner">
    <div>
      <p class="tri-ref-label" style="margin:0;">
        <span class="tri-ref-dot" style="background:{pred_color};"></span>
        Expected level (model)
      </p>
      <div class="tri-ref-value">KTAS {pred}</div>
    </div>
  </div>
  {n_card}
  {e_card}
  <div class="tri-legend">
    <strong>Legend:</strong>
    <strong style="color:#22c55e;">Green border</strong> normal triage (matches reference) ·
    <strong style="color:#16a34a;">Green badge</strong> correct ·
    <strong style="color:#ea580c;">Orange</strong> overtriage ·
    <strong style="color:#dc2626;">Red</strong> undertriage.
  </div>
</div>
"""


def render_results_pair(left_inner: str, right_inner: str) -> str:
    """Wrap left and right result panels in a flex row so both columns match height."""
    return f"""
<div class="tri-results-pair">
  <div class="tri-results-pair-inner">
    <div class="tri-results-pane tri-results-pane-left">{left_inner}</div>
    <div class="tri-results-pane tri-results-pane-right">{right_inner}</div>
  </div>
</div>
"""


def render_results_footnote() -> str:
    return f"""
<div class="tri-footnote">
  <strong style="color:{TEXT_PRIMARY};">Summary.</strong>
  The KTAS on the left is the <strong>model reference</strong> from patient inputs.
  The right panel shows nurse and physician assignments as correct/incorrect relative to that reference.
  Clinical decisions remain with the treating physician.
</div>
"""


# ── Main app ─────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="ED Triage Decision Support", page_icon="🏥", layout="wide")

    with st.sidebar:
        st.markdown("### Navigation")
        _section = st.radio(
            "Section",
            ["Triage", "Data visualization"],
            key="nav_section",
            label_visibility="collapsed",
        )
        st.caption("Charts use `data/data.csv` (no page switch needed).")

    if _section == "Data visualization":
        from src.dataviz_render import render_data_visualization

        render_data_visualization()
        return

    st.title("🏥 Emergency Department Triage Decision Support")
    st.caption(
        "**Step 1:** Enter patient data → model **KTAS prediction** (reference: expected level per model). "
        "**Step 2:** Enter nurse and physician KTAS → each is **compared to the model reference**."
    )
    st.caption(
        "📊 In the sidebar, choose **Data visualization** for charts of the training dataset."
    )

    st.warning(
        "⚠️ **For education and research only.** "
        "Not a substitute for real clinical decisions. Medical judgment must rest with a qualified clinician."
    )

    ktas_ok = Path("models/best_model.joblib").exists()
    if not ktas_ok:
        st.error(
            "KTAS model not found (`models/best_model.joblib`). Train first:\n\n"
            "```\npython run_pipeline.py\n```"
        )
        return

    st.markdown("---")
    st.subheader("1️⃣ Patient data (for KTAS prediction)")
    st.markdown("The model outputs an **estimated urgency level (KTAS 1–5)** from these fields.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Patient identity")

        age = st.number_input(
            "Age",
            min_value=0, max_value=120, value=45, step=1,
            help="Patient age",
        )
        sex = st.selectbox(
            "Sex",
            options=[1, 2],
            format_func=lambda x: "Female" if x == 1 else "Male",
            help="1 = Female, 2 = Male",
        )
        group = st.selectbox(
            "Hospital type",
            options=[1, 2],
            format_func=lambda x: "Local ED (3rd level)" if x == 1 else "Regional ED (4th level)",
            help="Emergency department tier",
        )
        arrival_mode = st.selectbox(
            "Arrival mode",
            options=[1, 2, 3, 4, 5, 6, 7],
            format_func=lambda x: {
                1: "Walk-in",
                2: "Public ambulance",
                3: "Private vehicle",
                4: "Private ambulance",
                5: "Other",
                6: "Other",
                7: "Other",
            }.get(x, str(x)),
            index=2,
            help="How the patient arrived at the ED",
        )
        injury = st.selectbox(
            "Injury present?",
            options=[1, 2],
            format_func=lambda x: "No" if x == 1 else "Yes",
            help="Trauma or injury-related visit",
        )
        patients_per_hour = st.slider(
            "Patients per hour (arrival hour load)",
            1, 20, 8,
            help="Total arrivals in the same hour (workload proxy)",
        )

    with col2:
        st.markdown("##### 🩺 Clinical assessment")

        _cc_idx = st.selectbox(
            "Chief complaint",
            options=range(len(CHIEF_COMPLAINT_DROPDOWN)),
            format_func=lambda i: CHIEF_COMPLAINT_DROPDOWN[i][0],
            index=0,
            help="Pick a category. The model uses the English phrase mapped to this choice.",
        )
        chief_complaint = CHIEF_COMPLAINT_DROPDOWN[_cc_idx][1]
        mental = st.selectbox(
            "Mental status",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Alert",
                2: "Verbal response",
                3: "Painful stimulus",
                4: "Unresponsive",
            }.get(x, str(x)),
            help="AVPU-style scale: Alert > Verbal > Pain > Unresponsive",
        )
        pain = st.selectbox(
            "Pain reported?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=1,
            help="Whether the patient reports pain",
        )
        if pain == 1:
            nrs_pain = st.slider(
                "Pain intensity (NRS 0–10)",
                0, 10, 3,
                help="0 = none, 10 = worst imaginable",
            )
        else:
            nrs_pain = None

    with col3:
        st.markdown("##### ❤️ Vitals")

        sbp = st.number_input(
            "Systolic BP (mmHg)",
            0, 300, 120,
            help="Typical normal range reference: ~90–120",
        )
        dbp = st.number_input(
            "Diastolic BP (mmHg)",
            0, 200, 80,
            help="Typical normal range reference: ~60–80",
        )
        hr = st.number_input(
            "Heart rate (bpm)",
            0, 300, 80,
            help="Beats per minute; typical ~60–100",
        )
        rr = st.number_input(
            "Respiratory rate (/min)",
            0, 60, 18,
            help="Breaths per minute; typical ~12–20",
        )
        bt = st.number_input(
            "Body temperature (°C)",
            30.0, 45.0, 36.5, 0.1,
            help="Typical ~36.1–37.2 °C",
        )
        saturation = st.number_input(
            "Oxygen saturation SpO₂ (%)",
            0, 100, 98,
            help="Typical ≥95% on room air",
        )

    st.markdown("---")
    st.subheader("2️⃣ Nurse and physician KTAS")
    st.markdown(
        "In KTAS, **1 = most urgent**, **5 = least urgent**. When you click **Run analysis**, "
        "the model runs first; nurse and physician levels are then **compared** to that prediction."
    )

    c_a, c_b = st.columns(2)
    with c_a:
        ktas_rn = st.selectbox(
            "Nurse KTAS level",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"KTAS {x}" + (" (higher urgency)" if x <= 3 else " (lower urgency)"),
            help="Triage nurse assigned urgency (1 = most urgent)",
        )
    with c_b:
        ktas_expert = st.selectbox(
            "Physician KTAS level",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"KTAS {x}" + (" (higher urgency)" if x <= 3 else " (lower urgency)"),
            index=3,
            help="Physician reference urgency level",
        )

    st.markdown("---")

    if st.button("🔍 Run analysis", type="primary", use_container_width=True):
        with st.spinner("Running KTAS model..."):
            model = load_model(Path("models/best_model.joblib"))
            ktas_result = predict_single(
                model,
                age=age, sex=sex, group=group,
                arrival_mode=arrival_mode, injury=injury,
                chief_complaint=chief_complaint,
                mental=mental, pain=pain, nrs_pain=nrs_pain,
                sbp=sbp, dbp=dbp, hr=hr, rr=rr, bt=bt,
                saturation=saturation,
                patients_per_hour=patients_per_hour,
            )

        pred = ktas_result["predicted_ktas"]
        nurse_style = triage_verdict_style(ktas_rn, pred)
        expert_style = triage_verdict_style(ktas_expert, pred)

        st.markdown("---")
        st.markdown("### Results")
        st.caption("Model prediction vs nurse and physician assignments")

        st.markdown(render_results_styles(), unsafe_allow_html=True)

        pred_color = KTAS_COLORS.get(pred, "#888")
        desc = KTAS_EN.get(pred, "")

        left_html = render_left_prediction_block(
            pred,
            desc,
            pred_color,
            ktas_result.get("probabilities"),
        )
        right_html = render_right_evaluation_block(
            pred,
            pred_color,
            ktas_rn,
            ktas_expert,
            nurse_style,
            expert_style,
        )
        st.markdown(
            render_results_pair(left_html, right_html),
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(render_results_footnote(), unsafe_allow_html=True)

    with st.expander("❓ Field reference"):
        st.markdown("""
| Field | Description |
|------|-------------|
| **Age** | Patient age |
| **Sex** | Female / Male |
| **Hospital type** | Local vs regional ED |
| **Arrival mode** | Walk-in, ambulance, private vehicle, etc. |
| **Injury** | Trauma-related visit |
| **Workload** | Arrivals in the same hour |
| **Chief complaint** | Category from dropdown |
| **Mental status** | Alert → verbal → pain → unresponsive |
| **Pain / NRS** | If pain, intensity 0–10 |
| **Vitals** | BP, HR, RR, temperature, SpO₂ |
| **Nurse KTAS** | Nurse level (1–5); compared to model |
| **Physician KTAS** | Physician level (1–5); compared to model |
""")

    with st.expander("📋 Complaint categories (model mapping)"):
        for keywords, category in CATEGORY_RULES:
            st.markdown(
                f"**{category}**: {', '.join(keywords[:8])}{'...' if len(keywords) > 8 else ''}"
            )


if __name__ == "__main__":
    main()
