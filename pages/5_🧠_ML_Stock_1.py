# pages/5_🧠_ML_Stock_1.py
# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Conclusion • Stock Buy Prediction", page_icon="✅", layout="wide")

# ---------- CSS (ใช้ .format() เลี่ยง f-string กับ {}) ----------
CSS = """
<style>
h1,h2,h3{letter-spacing:.2px}
.hr{height:1px;background:#edf2f7;margin:14px 0;border:0}
.card{border:1px solid #e8edf3;border-radius:14px;padding:14px;background:#ffffff}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;background:#eef6ff;border:1px solid #d6e7ff;color:#1e5a96;font-size:12px}
.kicker{font-size:12.5px;color:#6b7b85;margin-bottom:6px}
.block-title{font-weight:700;font-size:18px;margin:6px 0 2px}
.subtle{color:#5f6c72}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ROOT = Path(".")
MODEL_META = ROOT / "model_meta.json"

# ---------- รายชื่อฟีเจอร์ภาษาไทย ----------
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

# ---------- อ่าน meta ----------
feature_cols = list(FEATURE_THAI.keys())
suffix = ".BK"
horizon = 6
if MODEL_META.exists():
    meta = json.loads(MODEL_META.read_text(encoding="utf-8"))
    feature_cols = meta.get("feature_cols", feature_cols)
    suffix = meta.get("exchange_suffix", suffix)
    horizon = meta.get("forward_months", horizon)

# ---------- Section 1 ----------
st.title("โครงสร้างการนำเสนอ (ML Project: Stock Buy Prediction)")
st.markdown("""
**1. แนะนำแนวคิดโปรเจกต์ (Overview)**  
โครงการนี้เป็นระบบ Machine Learning สำหรับคัดกรองหุ้นไทย โดยใช้ปัจจัยพื้นฐาน (Fundamental Factors) เช่น ROE, Debt-to-Equity, Margin, OCF/CL เพื่อทำนายว่าหุ้นนั้นมีแนวโน้ม ‘น่าซื้อ’ หรือไม่ในอีก 6 เดือนข้างหน้า

- Flowchart: Input → Model → Output  
- Feature: อ้างอิงจาก model_meta.json
""")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 2 ----------
st.header("Section 2: การสร้าง Feature และ Target")
st.code('df["target"] = (df["forward_6m_return"] > 0).astype(int)', language="python")
st.markdown("- target = 1 → ราคามีแนวโน้มขึ้นใน {h} เดือน\n- target = 0 → ไม่ขึ้น (ลงหรือทรงตัว)".format(h=horizon))
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 3 (คงไว้ตามที่ขอ) ----------
st.header("Section 3: ฟีเจอร์ที่ใช้ (10 ตัว) — ภาษาไทย")
st.write("ตลาดที่ใช้: `{s}`  •  ขอบเขตการพยากรณ์: {h} เดือนล่วงหน้า".format(s=suffix, h=horizon))

cols = st.columns(2)
CARD_HTML = """
<div class="card">
  <div class="kicker">feature key</div>
  <div class="badge">{key}</div>
  <div class="block-title">{th_name}</div>
  <div class="subtle">{th_desc}</div>
</div>
"""
for i, key in enumerate(feature_cols):
    th_name, th_desc = FEATURE_THAI.get(key, (key, "-"))
    with cols[i % 2]:
        st.markdown(CARD_HTML.format(key=key, th_name=th_name, th_desc=th_desc), unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 4 ----------
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
st.markdown("- ใช้ RandomForestClassifier และจูน Threshold ให้ F1-Score ดีที่สุดบนชุดทดสอบ")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 5 ----------
st.header("Section 5: ผลลัพธ์ที่ได้ (Outputs)")
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("📘 dataset_financials_ttm.csv (ตัวอย่าง)")
    st.table(pd.DataFrame([{"ticker":"PTT","ROE":0.18,"ROA":0.09,"Debt/Equity":1.2,"Target":1}]))
with c2:
    st.subheader("📗 feature_importance.csv (ตัวอย่าง)")
    st.table(pd.DataFrame([
        {"Feature":"ROE_TTM","Importance":0.27},
        {"Feature":"Debt_to_equity","Importance":0.21},
        {"Feature":"OCF_to_CL","Importance":0.14},
        {"Feature":"Net_margin_ttm","Importance":0.12},
        {"Feature":"ROA_ttm","Importance":0.10},
    ]))
with c3:
    st.subheader("📙 predictions_latest.csv (ตัวอย่าง)")
    st.table(pd.DataFrame([
        {"Ticker":"SAPPE","Prob_Buy":0.88,"Pred":"✅"},
        {"Ticker":"AOT","Prob_Buy":0.72,"Pred":"✅"},
        {"Ticker":"IRPC","Prob_Buy":0.35,"Pred":"❌"},
    ]))
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ---------- Section 6 ----------
st.header("Section 6: สรุปการนำเสนอ (Conclusion Slide)")
st.markdown("""
- ✅ ดึงข้อมูลอัตโนมัติจาก Yahoo Finance  
- ✅ วิเคราะห์ปัจจัยพื้นฐาน 10 ตัว  
- ✅ ใช้ Random Forest จำแนก “หุ้นน่าซื้อ”  
- ✅ Export รายงานผลและความสำคัญของ Feature  
- 🚀 ต่อยอด: Dashboard & Alert System
""")
st.success("พร้อมใช้นำเสนอแล้ว ✅")
