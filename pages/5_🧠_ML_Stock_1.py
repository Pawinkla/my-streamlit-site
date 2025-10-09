# pages/5_🧠_ML_Stock_1.py
from pathlib import Path
import json
import streamlit as st

st.set_page_config(page_title="Conclusion • Stock Buy Prediction", page_icon="✅", layout="wide")

# ---------- Small style tweaks ----------
st.markdown("""
<style>
h1, h2, h3 { letter-spacing: .2px; }
.block-title {font-weight:700;font-size:20px;margin:6px 0 2px;}
.subtle { color:#5f6c72; font-size:14.5px; }
.hr { height:1px; background:linear-gradient(90deg,#e9eef2, #f6f8fa); border:0; margin:12px 0 16px; }
.card {
  border:1px solid #e8edf3; border-radius:14px; padding:14px 14px 12px;
  background:linear-gradient(180deg,#ffffff 0%, #fcfdff 100%);
  box-shadow: 0 1px 0 rgba(10,10,10,0.02);
}
.badge {display:inline-block; padding:2px 8px; font-size:12px; border-radius:999px; background:#eef6ff; border:1px solid #d6e7ff; color:#1e5a96;}
.kicker { font-size:13px; color:#6b7b85; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("สรุปโครงการ (Conclusion) ✅")
st.caption("ML Project: Stock Buy Prediction — นำเสนอแบบย่อ พร้อมอธิบายฟีเจอร์ภาษาไทย")

ROOT = Path(".")
MODEL_META = ROOT / "model_meta.json"

# ---------- Feature mapping (TH) ----------
FEATURE_THAI = {
    "revenue_growth_ttm":    ("การเติบโตของรายได้ (TTM)", "อัตราการเปลี่ยนแปลงของรายได้สะสมย้อนหลังเมื่อเทียบหน้าต่างก่อน"),
    "gross_margin_ttm":      ("อัตรากำไรขั้นต้น (TTM)", "ความสามารถทำกำไรจากยอดขายก่อนค่าใช้จ่ายดำเนินงาน"),
    "operating_margin_ttm":  ("อัตรากำไรจากการดำเนินงาน (TTM)", "ความสามารถทำกำไรจากธุรกิจหลักหลังหักค่าใช้จ่ายดำเนินงาน"),
    "net_margin_ttm":        ("อัตรากำไรสุทธิ (TTM)", "กำไรสุดท้ายต่อรายได้ทั้งหมด"),
    "roe_ttm":               ("ROE (TTM)", "กำไรสุทธิต่อส่วนของผู้ถือหุ้นเฉลี่ย—ชี้ประสิทธิภาพใช้ทุนผู้ถือหุ้น"),
    "roa_ttm":               ("ROA (TTM)", "กำไรสุทธิต่อสินทรัพย์เฉลี่ย—ประสิทธิภาพใช้สินทรัพย์สร้างกำไร"),
    "debt_to_equity":        ("Debt/Equity", "โครงสร้างทุน—หนี้รวมเทียบทุนผู้ถือหุ้น ยิ่งต่ำยิ่งปลอดภัย"),
    "ocf_to_cl":             ("OCF / Current Liabilities", "สภาพคล่องจากกระแสเงินสดดำเนินงานเทียบหนี้สินหมุนเวียน"),
    "interest_coverage":     ("Interest Coverage", "ความสามารถชำระดอกเบี้ย = EBIT / ดอกเบี้ยจ่าย"),
    "asset_turnover_ttm":    ("Asset Turnover (TTM)", "ประสิทธิภาพใช้สินทรัพย์สร้างยอดขาย"),
}

# ---------- Read meta (feature_cols / suffix / horizon) ----------
feature_cols = list(FEATURE_THAI.keys())
suffix = ".BK"
horizon = 6
if MODEL_META.exists():
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", feature_cols)
    suffix = meta.get("exchange_suffix", suffix)
    horizon = meta.get("forward_months", horizon)

# ---------- Section 1 : Overview ----------
st.header("Section 1: แนวคิดโปรเจกต์")
st.markdown("""
- ✅ ดึงข้อมูลอัตโนมัติจาก **Yahoo Finance** (งบรายไตรมาส/รายปี → คำนวณเป็น **TTM**)
- ✅ ใช้ฟีเจอร์ทางการเงิน **10 ตัว** เพื่อจำแนกว่า **“น่าซื้อ / ไม่น่าซื้อ”** ใน **{h} เดือนข้างหน้า** บนตลาด `{suf}`
- ✅ โมเดลหลัก: **Random Forest** พร้อมปรับ Threshold ตาม **F1-Score**
- ✅ ส่งออกผลลัพธ์เป็นไฟล์รายงาน (Predictions & Feature Importance) เพื่อใช้งานต่อ
""".format(h=horizon, suf=suffix))
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 2 : Code (สั้นกระชับเหมือนเดิม) ----------
st.header("Section 2: การสร้าง Feature และ Target")
st.code('df["target"] = (df["forward_6m_return"] > 0).astype(int)', language="python")
st.markdown("""
**อธิบาย:**  
- `target = 1` → ราคามีแนวโน้ม **ขึ้น** ใน {h} เดือน  
- `target = 0` → **ไม่ขึ้น** (ลงหรือทรงตัว)
""".format(h=horizon))
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 3 : ฟีเจอร์ที่ใช้ (ภาษาไทย) ----------
st.header("Section 3: ฟีเจอร์ที่ใช้ (10 ตัว) — ภาษาไทย")

# จัดการเป็นการ์ด 2 คอลัมน์
cols = st.columns(2)
for i, key in enumerate(feature_cols):
    th_name, th_desc = FEATURE_THAI.get(key, (key, "-"))
    with cols[i % 2]:
        st.markdown(
            f"""
<div class="card">
  <div class="kicker">feature key</div>
  <div class="badge">{key}</div>
  <div class="block-title">{th_name}</div>
  <div class="subtle">{th_desc}</div>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 4 : Pipeline & Training (ย่อ) ----------
st.header("Section 4: Pipeline & Training (ย่อ)")
st.code(
"""pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5,
        random_state=42, class_weight="balanced_subsample"
    ))
])
pipe.fit(X_train, y_train)""",
    language="python",
)
st.markdown("""
- ใช้ **RandomForestClassifier** เหมาะกับข้อมูลตัวเลขหลายมิติและทนทานต่อ outlier  
- ปรับ **Threshold** ด้วยฟังก์ชันจูนเพื่อให้ **F1-Score** ดีที่สุดบนชุดทดสอบ
""")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 5 : Conclusion ----------
st.header("Section 5: สรุปโครงการ")
st.markdown("""
- ✅ ดึงข้อมูลอัตโนมัติจาก **Yahoo Finance**
- ✅ วิเคราะห์ปัจจัยพื้นฐาน **10 ตัว** (ภาษาไทยด้านบน)
- ✅ ใช้ **Random Forest** จำแนกหุ้นน่าซื้อ
- ✅ ส่งออกผลและ **Feature Importance** เพื่ออธิบายโมเดล
- 🚀 **ต่อยอด**: ทำ Dashboard & Alert เมื่อความน่าซื้อเกินเกณฑ์
""")
st.success("พร้อมใช้นำเสนอทันที — โครงสร้างเรียบง่าย อ่านสบาย และสื่อสารชัดเจน ✨")
