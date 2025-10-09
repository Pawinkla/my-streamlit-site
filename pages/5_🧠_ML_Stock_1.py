# pages/5_🧠_ML_Stock_1.py
from pathlib import Path
import json
import pandas as pd
import streamlit as st

# ----------------------------
# ตั้งค่าหน้า
# ----------------------------
st.set_page_config(
    page_title="สรุปโครงการ • Fundamental ML",
    page_icon="✅",
    layout="wide",
)

st.title("สรุปโครงการ (Conclusion) ✅")
st.caption("โครงการ: คัดกรองหุ้นไทยด้วยปัจจัยพื้นฐานและ ML (Buy / Not Buy)")

ROOT = Path(".")
MODEL_META = ROOT / "model_meta.json"
FEATURE_IMP = ROOT / "feature_importance.csv"
PRED_LATEST = ROOT / "predictions_latest.csv"
DATASET_CSV = ROOT / "dataset_financials_ttm.csv"
MODEL_PKL = ROOT / "buy_model_pipeline.pkl"

# ----------------------------
# ส่วนที่ 1: สรุปใจความสำคัญ (Bullet แบบสไตล์สไลด์)
# ----------------------------
with st.container():
    st.subheader("สรุปการนำเสนอ")
    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown(
            """
**แนวคิดโดยรวม**
- ✅ **ดึงข้อมูลอัตโนมัติจาก Yahoo Finance** (งบรายไตรมาส/รายปี → แปลงเป็นค่าแบบ TTM)
- ✅ **วิเคราะห์ปัจจัยพื้นฐาน 10 ตัว** ที่สะท้อนกำไร ความสามารถทำเงิน และความแข็งแรงทางการเงิน
- ✅ **ใช้โมเดล Random Forest** เพื่อจำแนกหุ้นว่า “น่าซื้อ/ไม่น่าซื้อ”
- ✅ **ส่งออกไฟล์รายงาน** ทั้งตารางผลคาดการณ์ล่าสุด และความสำคัญของปัจจัย (Feature Importance)
- 🚀 **ต่อยอดได้ทันที**: ทำ Dashboard ติดตามสถานะรายสัปดาห์ และตั้งระบบแจ้งเตือน (Alert) เมื่อความน่าซื้อเกินเกณฑ์
            """
        )

    with col2:
        st.markdown("**ไฟล์สำคัญของโปรเจกต์ (ดาวน์โหลดได้ถ้ามีไฟล์ในโฟลเดอร์)**")
        files = [
            ("โมเดลที่เทรนแล้ว (.pkl)", MODEL_PKL),
            ("ข้อมูลเมตาของโมเดล (ฟีเจอร์/ตลาด/ระยะเวลา)", MODEL_META),
            ("ความสำคัญของปัจจัย (Feature Importance)", FEATURE_IMP),
            ("ผลคาดการณ์ล่าสุดต่อหุ้น (Predictions)", PRED_LATEST),
            ("ชุดข้อมูลที่ใช้สอนโมเดล (Dataset TTM)", DATASET_CSV),
        ]
        for label, path in files:
            if path.exists():
                st.download_button(
                    label=f"⬇️ ดาวน์โหลด {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/octet-stream",
                )
            else:
                st.write(f"▫️ {label}: _ยังไม่พบไฟล์ `{path.name}`_")

st.divider()

# ----------------------------
# ส่วนที่ 2: รายละเอียดโมเดลแบบภาษาไทย (อิง model_meta.json)
# ----------------------------
st.subheader("รายละเอียดโมเดลโดยย่อ (ภาษาไทย)")

feature_map_th = {
    "revenue_growth_ttm": "การเติบโตของรายได้ (TTM เทียบหน้าต่างก่อนหน้า)",
    "gross_margin_ttm": "อัตรากำไรขั้นต้น (Gross Margin, TTM)",
    "operating_margin_ttm": "อัตรากำไรจากการดำเนินงาน (Operating Margin, TTM)",
    "net_margin_ttm": "อัตรากำไรสุทธิ (Net Margin, TTM)",
    "roe_ttm": "ROE (กำไรสุทธิต่อส่วนของผู้ถือหุ้นเฉลี่ย, TTM)",
    "roa_ttm": "ROA (กำไรสุทธิต่อสินทรัพย์เฉลี่ย, TTM)",
    "debt_to_equity": "อัตราส่วนหนี้สินรวมต่อส่วนของผู้ถือหุ้น",
    "ocf_to_cl": "กระแสเงินสดจากการดำเนินงาน / หนี้สินหมุนเวียน",
    "interest_coverage": "ความสามารถชำระดอกเบี้ย (EBIT / ดอกเบี้ยจ่าย)",
    "asset_turnover_ttm": "ประสิทธิภาพการใช้สินทรัพย์ (ยอดขาย/สินทรัพย์เฉลี่ย, TTM)",
}

if MODEL_META.exists():
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", [])
    suffix = meta.get("exchange_suffix", "")
    fwd = meta.get("forward_months", 6)

    st.write(f"**ตลาดที่ใช้:** `{suffix}`  •  **ขอบเขตการพยากรณ์:** {fwd} เดือนล่วงหน้า")

    # แสดงเป็น "ตารางคำอธิบายฟีเจอร์ภาษาไทย" แทนการโชว์เป็น pill
    rows = []
    for f in feature_cols:
        rows.append({
            "ชื่อฟีเจอร์ (โค้ด)": f,
            "ความหมาย (ไทย)": feature_map_th.get(f, "-"),
        })
    df_feat = pd.DataFrame(rows)
    st.dataframe(df_feat, use_container_width=True)
else:
    st.info("ยังไม่พบ `model_meta.json` (กรุณาเทรนโมเดลเพื่อสร้างไฟล์นี้ก่อน)")

st.divider()

# ----------------------------
# ส่วนที่ 3: ความสำคัญของปัจจัย (Feature Importance)
# ----------------------------
st.subheader("ความสำคัญของปัจจัย (Feature Importance)")
def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

fi = safe_read_csv(FEATURE_IMP)
if fi is not None:
    # กรณี export มาจาก Series จะเป็นคอลัมน์ 0,1 → รีเนมให้อ่านง่าย
    if set(fi.columns.astype(str)) == {"0", "1"}:
        fi.columns = ["feature", "importance"]
    if {"feature", "importance"}.issubset(fi.columns):
        fi = fi.sort_values("importance", ascending=False)
        st.bar_chart(data=fi.set_index("feature"))
        st.dataframe(fi.reset_index(drop=True), use_container_width=True)
    else:
        st.info("รูปแบบไฟล์ feature_importance.csv ยังไม่ตรงคอลัมน์ที่คาดหวัง (feature, importance)")
else:
    st.info("ยังไม่พบไฟล์ `feature_importance.csv`")

st.divider()

# ----------------------------
# ส่วนที่ 4: ผลคาดการณ์ล่าสุดแบบสรุป (ถ้ามีไฟล์)
# ----------------------------
st.subheader("ผลคาดการณ์ล่าสุด (Top Picks)")
pred = safe_read_csv(PRED_LATEST)
if pred is not None:
    rename_map = {
        "ticker": "สัญลักษณ์",
        "asof": "งบ ณ วันที่",
        "proba_buy": "ความน่าซื้อ (%)",
        "pred": "ผลการประเมิน",
    }
    pred_disp = pred.rename(columns=rename_map).copy()
    if "ความน่าซื้อ (%)" in pred_disp.columns:
        pred_disp["ความน่าซื้อ (%)"] = (pred_disp["ความน่าซื้อ (%)"] * 100).round(1)
    if "ผลการประเมิน" in pred_disp.columns:
        pred_disp["ผลการประเมิน"] = pred_disp["ผลการประเมิน"].map({1: "น่าซื้อ ✅", 0: "ไม่น่าซื้อ ❌"})
    st.dataframe(
        pred_disp.sort_values("ความน่าซื้อ (%)", ascending=False).head(15),
        use_container_width=True,
    )
else:
    st.info("ยังไม่พบไฟล์ `predictions_latest.csv`")

st.divider()

# ----------------------------
# ส่วนที่ 5: แนวทางต่อยอด (Next Steps)
# ----------------------------
st.subheader("แนวทางต่อยอด (Next Steps)")
st.markdown(
    """
- **Dashboard ติดตามผล**: หน้าแสดง Top Picks, แนวโน้มทศนิยม, และกราฟเปรียบเทียบกับดัชนีอ้างอิง
- **Alert System**: ตั้งเกณฑ์ `ความน่าซื้อ ≥ threshold` แล้วแจ้งเตือนผ่าน Line/Telegram/Email
- **Model Lifecycle**: เทรนซ้ำรายเดือน, เก็บ Metric (F1, ROC-AUC, Precision/Recall), และปรับ Threshold อัตโนมัติ
    """
)

st.success("หน้านี้สรุปเป็นภาษาไทยครบถ้วน พร้อมใช้นำเสนอแล้ว ✨")
