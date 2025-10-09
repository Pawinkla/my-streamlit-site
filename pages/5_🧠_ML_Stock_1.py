# pages/5_🧠_ML_Stock_1.py
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Conclusion • Stock Buy Prediction", page_icon="✅", layout="wide")

# ====== Style (เล็กน้อยให้ดูคลีน) ======
st.markdown("""
<style>
h1,h2,h3{letter-spacing:.2px}
.hr{height:1px;background:#edf2f7;margin:14px 0;border:0}
.card{border:1px solid #e8edf3;border-radius:14px;padding:14px;background:#fff}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;background:#eef6ff;border:1px solid #d6e7ff;color:#1e5a96;font-size:12px}
.kicker{font-size:12.5px;color:#6b7b85;margin-bottom:6px}
.block-title{font-weight:700;font-size:18px;margin:6px 0 2px}
.subtle{color:#5f6c72}
</style>
""", unsafe_allow_html=True)

ROOT = Path(".")
MODEL_META = ROOT / "model_meta.json"

# ====== Meta (feature list / suffix / horizon) ======
FEATURE_THAI = {
    "revenue_growth_ttm": ("การเติบโตของรายได้ (TTM)", "อัตราการเปลี่ยนแปลงของรายได้สะสมย้อนหลังเมื่อเทียบงวดก่อนหน้า"),
    "gross_margin_ttm":   ("อัตรากำไรขั้นต้น (TTM)", "กำไรขั้นต้นต่อรายได้รวม"),
    "operating_margin_ttm":("อัตรากำไรจากการดำเนินงาน (TTM)", "กำไรจากธุรกิจหลักหลังหักค่าใช้จ่ายดำเนินงาน"),
    "net_margin_ttm":     ("อัตรากำไรสุทธิ (TTM)", "กำไรสุทธิต่อรายได้รวม"),
    "roe_ttm":            ("ROE (TTM)", "กำไรสุทธิต่อส่วนของผู้ถือหุ้นเฉลี่ย"),
    "roa_ttm":            ("ROA (TTM)", "กำไรสุทธิต่อสินทรัพย์เฉลี่ย"),
    "debt_to_equity":     ("Debt/Equity", "หนี้สินรวมเทียบส่วนของผู้ถือหุ้น"),
    "ocf_to_cl":          ("OCF / Current Liabilities", "กระแสเงินสดจากการดำเนินงานเทียบหนี้สินหมุนเวียน"),
    "interest_coverage":  ("Interest Coverage", "EBIT / ดอกเบี้ยจ่าย"),
    "asset_turnover_ttm": ("Asset Turnover (TTM)", "ประสิทธิภาพใช้สินทรัพย์สร้างยอดขาย"),
}
feature_cols = list(FEATURE_THAI.keys())
suffix = ".BK"
horizon = 6
if MODEL_META.exists():
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", feature_cols)
    suffix = meta.get("exchange_suffix", suffix)
    horizon = meta.get("forward_months", horizon)

# ====== โครงสร้างการนำเสนอ (เหมือน PDF) ======
st.title("โครงสร้างการนำเสนอ (ML Project: Stock Buy Prediction)")
st.markdown("""
**1. แนะนำแนวคิดโปรเจกต์ (Overview)**  
“โครงการนี้เป็นระบบ Machine Learning สำหรับคัดกรองหุ้นไทย โดยใช้ปัจจัยพื้นฐาน (Fundamental Factors) เช่น ROE, Debt-to-Equity, Margin, OCF/CL เพื่อทำนายว่าหุ้นนั้นมีแนวโน้ม ‘น่าซื้อ’ หรือไม่ในอีก 6 เดือนข้างหน้า”  :contentReference[oaicite:2]{index=2}

- Flowchart (Input → Model → Output)  
- Feature ตัวอย่างจาก `model_meta.json` เช่น ["ROE","ROA","Gross Margin","Debt/Equity","OCF/CL", ...]  :contentReference[oaicite:3]{index=3}
""")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== Section 2: โค้ดส่วนสำคัญ (เหมือน PDF) ======
st.header("Section 2: แสดงโค้ดส่วนสำคัญ (จาก Auto_Fundamental_fixed_v7.py)")

st.subheader("🔹 Section 1: การสร้าง Dataset")
st.code(
"""def build_dataset():
    for tk in TICKERS:
        t = yf.Ticker(tk + ".BK")
        price_hist = t.history(period="max")
        q_rows = extract_samples_quarterly(t, price)""",
language="python")
st.markdown("อธิบาย: ดึงข้อมูลจาก Yahoo Finance สำหรับหุ้นในตลาดไทย แล้วคำนวณตัวชี้วัดเป็น **TTM** (Trailing 12 Months). :contentReference[oaicite:4]{index=4}")

st.subheader("🔹 Section 2: การสร้าง Feature และ Target")
st.code('df["target"] = (df["forward_6m_return"] > 0).astype(int)', language="python")
st.markdown("อธิบาย: Target = 1 หมายถึงราคาหุ้นขึ้นในอีก 6 เดือน, Target = 0 หมายถึงราคาลงหรือเท่าเดิม. :contentReference[oaicite:5]{index=5}")

st.subheader("🔹 Section 3: การสร้าง Pipeline และเทรนโมเดล")
st.code(
"""pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5,
        random_state=42, class_weight="balanced_subsample"
    ))
])
pipe.fit(X_train, y_train)""", language="python")
st.markdown("อธิบาย: ใช้ **RandomForestClassifier** เหมาะกับข้อมูลตัวเลขหลายมิติ (financial ratios) และช่วยลด **overfitting**. :contentReference[oaicite:6]{index=6}")

st.subheader("🔹 Section 4: การปรับ Threshold")
st.code(
"""thr, best_f1 = tune_threshold(y_test, y_prob)
print(f"Best threshold by F1: {thr:.3f}")""", language="python")
st.markdown("อธิบาย: เลือก threshold ที่ให้ค่า **F1-Score** ดีที่สุด เพื่อให้แม่นทั้งฝั่ง Buy และ Not-Buy. :contentReference[oaicite:7]{index=7}")

st.subheader("🔹 Section 5: Export & Result")
st.code(
"""dump(pipe, "buy_model_pipeline.pkl")
out.to_csv("predictions_latest.csv")""", language="python")
st.markdown("อธิบาย: บันทึกโมเดล `.pkl` และส่งออก “ผลคาดการณ์ล่าสุด” เป็น `predictions_latest.csv`. :contentReference[oaicite:8]{index=8}")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== Section 3: ฟีเจอร์ที่ใช้ (10 ตัว) — ภาษาไทย ======
st.header("Section 3: ฟีเจอร์ที่ใช้ (10 ตัว) — ภาษาไทย")
st.write(f"ตลาดที่ใช้: `{suffix}`  •  ขอบเขตการพยากรณ์: {horizon} เดือนล่วงหน้า :contentReference[oaicite:9]{index=9}")

cols = st.columns(2)
for i, key in enumerate(feature_cols):
    th_name, th_desc = FEATURE_THAI.get(key, (key, "-"))
    with cols[i % 2]:
        st.markdown(f"""
<div class="card">
  <div class="kicker">feature key</div>
  <div class="badge">{key}</div>
  <div class="block-title">{th_name}</div>
  <div class="subtle">{th_desc}</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== Section 4: ผลลัพธ์ที่ได้ (Outputs) — เหมือน PDF ======
st.header("Section 4: ผลลัพธ์ที่ได้ (Outputs)")
st.markdown("ให้โชว์ **3 ตารางหลัก** (ตัวอย่างเหมือนในสไลด์) :contentReference[oaicite:10]{index=10}")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("📘 1) dataset_financials_ttm.csv")
    st.markdown("**ตัวอย่างข้อมูลเทรน**")
    st.table(pd.DataFrame(
        [{"ticker":"PTT","ROE":0.18,"ROA":0.09,"Debt/Equity":1.2,"Target":1}]
    ))  # filecite: sample row
with c2:
    st.subheader("📗 2) feature_importance.csv")
    st.markdown("**Top 5 ปัจจัยสำคัญที่สุด**")
    st.table(pd.DataFrame(
        [
            {"Feature":"ROE_TTM","Importance":0.27},
            {"Feature":"Debt_to_equity","Importance":0.21},
            {"Feature":"OCF_to_CL","Importance":0.14},
            {"Feature":"Net_margin_ttm","Importance":0.12},
            {"Feature":"ROA_ttm","Importance":0.10},
        ]
    ))  # :contentReference[oaicite:11]{index=11}
with c3:
    st.subheader("📙 3) predictions_latest.csv")
    st.markdown("**ตัวอย่าง Output**")
    st.table(pd.DataFrame(
        [
            {"Ticker":"SAPPE","Prob_Buy":0.88,"Pred":"✅"},
            {"Ticker":"AOT","Prob_Buy":0.72,"Pred":"✅"},
            {"Ticker":"IRPC","Prob_Buy":0.35,"Pred":"❌"},
        ]
    ))  # :contentReference[oaicite:12]{index=12}

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== Section 5: การรันและ Deployment — เหมือน PDF ======
st.header("Section 5: การรันและ Deployment")
st.code("pip install -r requirements.txt", language="bash")   # :contentReference[oaicite:13]{index=13}
st.code("streamlit run app.py", language="bash")              # :contentReference[oaicite:14]{index=14}

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ====== Section 6: สรุปการนำเสนอ (Conclusion Slide) — เหมือน PDF ======
st.header("Section 6: สรุปการนำเสนอ (Conclusion Slide)")
st.markdown("""
- ✅ ดึงข้อมูลอัตโนมัติจาก **Yahoo Finance**  
- ✅ วิเคราะห์ **ปัจจัยพื้นฐาน 10 ตัว**  
- ✅ ใช้ **Random Forest** จำแนก “หุ้นน่าซื้อ”  
- ✅ **Export** รายงานผลและความสำคัญของ Feature  
""")  # :contentReference[oaicite:15]{index=15}

st.success("หน้านี้ทำให้ตรงกับเอกสารพรีเซนต์แล้วครบทุกส่วน • พร้อมใช้นำเสนอทันที")
