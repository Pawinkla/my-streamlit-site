# pages/5_🧠_ML_Stock_1.py
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Conclusion • Fundamental ML",
    page_icon="✅",
    layout="wide",
)

st.title("สรุปโครงการ (Conclusion) ✅")
st.caption("Fundamental ML • Stock Buy/Not-Buy Classification")

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(".")
MODEL_META = ROOT / "model_meta.json"
FEATURE_IMP = ROOT / "feature_importance.csv"
PRED_LATEST = ROOT / "predictions_latest.csv"
DATASET_CSV = ROOT / "dataset_financials_ttm.csv"
MODEL_PKL = ROOT / "buy_model_pipeline.pkl"

# ----------------------------
# Helper
# ----------------------------
def safe_read_csv(path: Path, n=8):
    try:
        df = pd.read_csv(path)
        return df if n is None else df.head(n)
    except Exception as e:
        st.info(f"ยังไม่พบไฟล์ `{path.name}` หรือเปิดไม่ได้ ({e})")
        return None

def pill(text: str):
    return f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;background:#eef6ff;border:1px solid #d6e7ff;margin:2px 6px 2px 0'>{text}</span>"

# ----------------------------
# Summary bullets
# ----------------------------
with st.container():
    st.subheader("สรุปการนำเสนอ")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(
            """
- ✅ **ดึงข้อมูลอัตโนมัติจาก Yahoo Finance** (quarterly/annual statements → TTM)
- ✅ **วิเคราะห์ปัจจัยพื้นฐาน 10 ตัว** (เช่น ROE, ROA, Margin, Debt/Equity, OCF/CL)
- ✅ **ใช้ Random Forest จำแนก “หุ้นน่าซื้อ”**
- ✅ **Export รายงานผลและความสำคัญของ Feature** (`feature_importance.csv`, `predictions_latest.csv`)
- 🚀 **ต่อยอดได้**: ทำ Dashboard ติดตามผล & ระบบแจ้งเตือน (Alert) ตามเกณฑ์ความน่าซื้อ
            """
        )

    with col2:
        st.markdown("**ไฟล์สำคัญของโปรเจกต์**")
        files = [
            ("โมเดล", MODEL_PKL),
            ("เมตาโมเดล", MODEL_META),
            ("Feature Importance", FEATURE_IMP),
            ("ผลคาดการณ์ล่าสุด", PRED_LATEST),
            ("Dataset (TTM)", DATASET_CSV),
        ]
        for label, path in files:
            if path.exists():
                st.download_button(
                    label=f"⬇️ ดาวน์โหลด {label} ({path.name})",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                )
            else:
                st.write(f"▫️ {label}: _ยังไม่พบไฟล์ {path.name}_")

# ----------------------------
# Model meta + feature list
# ----------------------------
st.divider()
st.subheader("รายละเอียดโมเดลโดยย่อ")

if MODEL_META.exists():
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", [])
    suffix = meta.get("exchange_suffix", "")
    fwd = meta.get("forward_months", 6)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write(f"**Exchange:** `{suffix}` • **Horizon:** {fwd} เดือนล่วงหน้า")
        st.write("**Feature ที่ใช้ (10 ตัว):**", unsafe_allow_html=True)
        st.markdown(
            " ".join([pill(x) for x in feature_cols]),
            unsafe_allow_html=True,
        )
    with c2:
        st.code(
            "feature_cols = [\n  " + ",\n  ".join([f'"{c}"' for c in feature_cols]) + "\n]",
            language="python",
        )
else:
    st.info("ยังไม่พบ `model_meta.json` — รันสคริปต์เทรนเพื่อสร้างไฟล์นี้ก่อน")

# ----------------------------
# Feature importance
# ----------------------------
st.divider()
st.subheader("ความสำคัญของปัจจัย (Feature Importance)")

fi = safe_read_csv(FEATURE_IMP, n=None)
if fi is not None and {"0", "1"}.issubset(set(fi.columns.astype(str))):
    # กรณีไฟล์เป็น series.to_csv() จะได้คอลัมน์ชื่อ 0,1
    fi.columns = ["feature", "importance"]
if fi is not None and {"feature", "importance"}.issubset(fi.columns):
    fi = fi.sort_values("importance", ascending=False)
    st.bar_chart(data=fi.set_index("feature"))
    st.dataframe(fi.reset_index(drop=True), use_container_width=True)
else:
    st.info("ยังไม่มีข้อมูลความสำคัญของ feature")

# ----------------------------
# Latest predictions
# ----------------------------
st.divider()
st.subheader("ผลคาดการณ์ล่าสุด (Top picks)")

pred = safe_read_csv(PRED_LATEST, n=None)
if pred is not None:
    # แปลงคอลัมน์ตามชื่อที่สคริปต์เทรน export
    rename_map = {
        "ticker": "Ticker",
        "asof": "งบ ณ วันที่",
        "proba_buy": "โอกาสน่าซื้อ (%)",
        "pred": "ผลการประเมิน",
    }
    pred_disp = pred.rename(columns=rename_map).copy()
    if "โอกาสน่าซื้อ (%)" in pred_disp.columns:
        pred_disp["โอกาสน่าซื้อ (%)"] = (pred_disp["โอกาสน่าซื้อ (%)"] * 100).round(1)
    if "ผลการประเมิน" in pred_disp.columns:
        pred_disp["ผลการประเมิน"] = pred_disp["ผลการประเมิน"].map({1: "น่าซื้อ ✅", 0: "ไม่น่าซื้อ ❌"})
    st.dataframe(
        pred_disp.sort_values("โอกาสน่าซื้อ (%)", ascending=False).head(15),
        use_container_width=True,
    )
else:
    st.info("ยังไม่พบ `predictions_latest.csv`")

# ----------------------------
# Next steps (callouts)
# ----------------------------
st.divider()
st.subheader("Next Steps / การต่อยอด")
st.markdown(
    """
- 📊 **Dashboard ติดตามผล**: สร้างหน้า Streamlit แสดง Top Picks, กราฟ Performance เทียบ SET, และลิสต์หุ้นที่เปลี่ยนสถานะ
- 🔔 **Alert System**: ตั้งเงื่อนไขแจ้งเตือนเมื่อ `proba_buy ≥ เกณฑ์ที่ตั้ง` แล้วส่งไปทาง Line/Telegram หรือ Email
- 🧪 **Model Lifecycle**: เทรนแบบกำหนดรอบ (เช่น รายเดือน) และ Log metric (F1, ROC-AUC, Precision/Recall) เพื่อปรับ threshold อย่างเป็นระบบ
    """
)

st.success("🎉 พร้อมสำหรับการนำเสนอแล้ว!  หน้า ‘Conclusion’ นี้สรุปทั้งแนวคิด วิธีทำ และผลลัพธ์สำคัญครบถ้วน")
