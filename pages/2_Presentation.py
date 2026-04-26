"""
Project presentation page:
- What was done in the notebook
- Why each feature exists
- How the app works
- A concrete patient example (nurse vs AI vs expert)
"""

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
st.caption("Notebook'ta yapılan tüm sürecin sade anlatımı + uygulamanın gerçek kullanım değeri")

st.markdown(
    """
Bu sayfa, projeyi hiç bilmeyen birinin bile anlayacağı şekilde şu soruları yanıtlar:
- Veri neydi, nasıl temizlendi?
- Feature engineering neden yapıldı?
- Model nasıl eğitildi ve nasıl ölçüldü?
- Uygulama sahada nasıl kullanılacak?
- Hemşire hatasını azaltma açısından pratik katkı ne olabilir?
"""
)

st.markdown("---")
st.header("1) Problem ve business logic")
st.markdown(
    """
**İş problemi:** Acil serviste triyaj seviyesi hatalı verildiğinde:
- **Overtriage (gereğinden daha acil):** kaynak israfı ve akış bozulması,
- **Undertriage (gereğinden az acil):** hasta güvenliği riski oluşur.

**Business logic:**  
Model, hasta girişindeki klinik verilerden **referans KTAS** önerir.  
Sistem, hemşire KTAS ile modeli karşılaştırarak **erken ikinci görüş** sağlar.
"""
)

st.markdown("---")
st.header("2) Notebook'ta ne yapıldı? (uçtan uca)")
st.markdown(
    """
### A. Veri hazırlama
1. `data/data.csv` yüklendi (`;` ayırıcı, ondalık düzeltmeleri).  
2. Eksik/bozuk değerler (`??`, bozuk `NRS_pain`) temizlendi.  
3. Kategorik alanlar normalize edildi (Sex, Injury, Mental, Arrival mode vb.).

### B. Keşifsel analiz (EDA)
1. KTAS dağılımı (hemşire vs uzman),  
2. mistriage dağılımı (normal/over/under),  
3. yaş, vitaller, kalış süresi ve yoğunluk dağılımları.

### C. Feature engineering
1. Chief complaint metni normalize edilip kategoriye çevrildi.  
2. Vitals + demografi + operasyonel alanlardan model girdileri kuruldu.  
3. Gerekli encoding/ölçekleme yapıldı.

### D. Modelleme
1. Çok sınıflı sınıflandırma ile KTAS tahmini eğitildi,  
2. Hold-out test/CV ile performans kontrol edildi,  
3. En iyi pipeline `models/best_model.joblib` olarak kaydedildi.

### E. Ürünleştirme
1. Streamlit app ile form tabanlı tahmin ekranı,  
2. Data Visualization sayfası ile veri hikayesi,  
3. Bu Presentation sayfası ile teknik + iş anlatımı.
"""
)

st.markdown("---")
st.header("3) Neden bu değişkenler var? (hiç bilmeyen için)")
feature_df = pd.DataFrame(
    [
        ("Yaş, cinsiyet", "Temel risk profili ve popülasyon farkları"),
        ("Geliş şekli (ambulans/yürüme)", "Vaka ciddiyetinin dolaylı göstergesi"),
        ("Mental durum", "Acil ciddiyeti için güçlü klinik sinyal"),
        ("Ağrı + NRS", "Semptom şiddeti"),
        ("SBP/DBP/HR/RR/BT/SpO2", "Hayati bulgular; triyajın çekirdeği"),
        ("Saatlik hasta sayısı", "Yoğunluk/operasyonel baskı etkisi"),
        ("Chief complaint", "Semptom bağlamı (ör. chest pain vs minor wound)"),
        ("Nurse vs Expert KTAS", "Hata analizi ve kalite ölçümü"),
    ],
    columns=["Alan", "Neden kullanılır?"],
)
st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.header("4) Uygulama nasıl kullanılır?")
st.markdown(
    """
1. **App** sayfasında hasta formunu doldur.  
2. **Run analysis** ile AI'nin önerdiği KTAS seviyesini al.  
3. Hemşire ve doktor KTAS seviyelerini girince sistem:
   - hemşireyi modele göre karşılaştırır (doğru / over / under),
   - doktoru modele göre karşılaştırır.
4. Sonuçlar karar vericiye “erken uyarı” sağlar; son karar yine klinisyendedir.
"""
)

st.markdown("---")
st.header("5) Gerçek vaka örneği: hemşire vs AI")
work, sample, metrics, _ = _analysis_cache()

if metrics is None:
    st.warning("Model dosyası bulunamadı (`models/best_model.joblib`). Bu bölüm için önce modeli eğitmelisiniz.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Kayıt sayısı", f"{metrics['n']}")
    c2.metric("Hemşire-Uzman uyumu", f"{metrics['nurse_agreement']*100:.1f}%")
    c3.metric("AI-Uzman uyumu", f"{metrics['ai_agreement']*100:.1f}%")

    st.markdown(
        f"""
**Potansiyel etki (dataset içi retrospektif):**
- Hemşirenin uzmanla uyuşmadığı vaka: **{metrics['nurse_errors']}**
- Bu hataların içinde AI'nin uzmanı tuttuğu vaka: **{metrics['potentially_corrected']}**
- Potansiyel yakalama oranı: **{metrics['potentially_corrected_ratio']*100:.1f}%**
"""
    )

    st.subheader("Örnek kişi")
    ex = {
        "Chief complaint": sample.get("Chief_complain"),
        "Age": int(sample.get("Age")),
        "Mental": int(sample.get("Mental")),
        "Pain": int(sample.get("Pain")),
        "SBP/DBP": f"{sample.get('SBP')}/{sample.get('DBP')}",
        "HR": sample.get("HR"),
        "RR": sample.get("RR"),
        "SpO2": sample.get("Saturation"),
        "Hemşire KTAS": int(sample.get("KTAS_RN_num")),
        "Uzman KTAS": int(sample.get("KTAS_expert_num")),
        "AI KTAS": int(sample.get("ai_ktas")),
    }
    st.json(ex)

    nurse = int(sample.get("KTAS_RN_num"))
    expert = int(sample.get("KTAS_expert_num"))
    ai = int(sample.get("ai_ktas"))

    if nurse != expert and ai == expert:
        st.success(
            "Bu örnekte hemşire seviyesi uzmanla uyuşmuyor; AI ise uzman seviyesini tutturuyor. "
            "Yani sistem, bu vakada olası triyaj hatasını erken yakalamaya yardımcı olabilirdi."
        )
    elif nurse == expert:
        st.info("Bu örnekte hemşire zaten doğru seviyeyi vermiş (uzmanla uyumlu).")
    else:
        st.warning("Bu örnekte AI de uzmanı tutturamamış. Bu da modelin sınırlılıklarını gösterir.")

st.markdown("---")
st.header("6) Sınırlılıklar ve doğru kullanım")
st.markdown(
    """
- Bu sistem **karar destek** içindir, **karar verici değildir**.  
- Model dataset'teki örüntüleri öğrenir; yeni dağılımlarda performans düşebilir.  
- Klinik bağlam, muayene ve hekim kararı her zaman önceliklidir.  
- Düzenli izleme, yeniden eğitim ve kalite metrikleri gerekir.
"""
)

