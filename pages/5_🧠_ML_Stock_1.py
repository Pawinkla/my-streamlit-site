# pages/5_🧠_ML_Stock_1.py
import streamlit as st

st.set_page_config(page_title="Conclusion • Stock Buy Prediction", page_icon="✅", layout="wide")

# -----------------------------
# สรุปโครงการ (ตรงตามไฟล์ PDF)
# -----------------------------
st.title("สรุปโครงการ (Conclusion) ✅")
st.caption("ML Project: Stock Buy Prediction — นำเสนอแบบย่อ เหมือนเอกสารสรุป")

st.header("1) แนะนำแนวคิดโปรเจกต์ (Overview)")
st.markdown(
    """
โปรเจกต์นี้คือระบบ **Machine Learning** สำหรับ **คัดกรองหุ้นไทย** โดยใช้ **ปัจจัยพื้นฐาน (Fundamental Factors)** 
เช่น **ROE, Debt-to-Equity, Margin, OCF/CL** เพื่อทำนายว่า **หุ้นนั้นมีแนวโน้ม “น่าซื้อ” หรือไม่ในอีก 6 เดือนข้างหน้า**  

**โครงภาพรวม (Flow):**  
- **Input:** งบการเงินรายไตรมาส/รายปี → แปลงเป็นค่าแบบ TTM  
- **Model:** สร้างฟีเจอร์ทางการเงิน 10 รายการ → เทรนโมเดลจำแนกผล  
- **Output:** ผล “น่าซื้อ / ไม่น่าซื้อ” และความน่าจะเป็น พร้อมรายงานสรุป
"""
)

st.header("2) โค้ดส่วนสำคัญ (จาก Auto_Fundamental_fixed_v7.py)")
st.subheader("🔹 Section 1: การสร้าง Dataset")
st.code(
    """def build_dataset():
    for tk in TICKERS:
        t = yf.Ticker(tk + ".BK")
        price_hist = t.history(period="max")
        price = price_hist.get("Close")
        q_rows = extract_samples_quarterly(t, price)""",
    language="python",
)
st.markdown(
    """
**อธิบาย:** ดึงข้อมูลจาก Yahoo Finance สำหรับหุ้นไทยหลายตัว และคำนวณตัวชี้วัด (ROE, Margin, ฯลฯ) เป็นข้อมูลแบบ **TTM**  
หากงบรายไตรมาสไม่พอ จะถอยไปใช้งบรายปีเพื่อคำนวณฟีเจอร์
"""
)

st.subheader("🔹 Section 2: การสร้าง Feature และ Target")
st.code('df["target"] = (df["forward_6m_return"] > 0).astype(int)', language="python")
st.markdown(
    """
**อธิบาย:**  
- `target = 1` หมายถึง ราคาหุ้น **มีแนวโน้มขึ้น** ในอีก 6 เดือน  
- `target = 0` หมายถึง **ไม่ขึ้น** (ลงหรือทรงตัว)
"""
)

st.subheader("🔹 Section 3: การสร้าง Pipeline และเทรนโมเดล")
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
st.markdown(
    """
**อธิบาย:** ใช้ **RandomForestClassifier** เหมาะกับข้อมูลเชิงตัวเลขหลายมิติ และช่วยลดโอกาส **overfitting**
"""
)

st.subheader("🔹 Section 4: การปรับ Threshold ให้เหมาะสม")
st.code(
    """thr, best_f1 = tune_threshold(y_test, y_prob)
print(f"Best threshold by F1: {thr:.3f}")""",
    language="python",
)
st.markdown("**อธิบาย:** ปรับค่าขอบการตัดสินใจ (threshold) เพื่อเพิ่ม **F1-Score** ให้สมดุลทั้ง Precision/Recall")

st.subheader("🔹 Section 5: Export & Result")
st.code(
    """dump(pipe, "buy_model_pipeline.pkl")        # บันทึกโมเดล
out.to_csv("predictions_latest.csv", index=False)  # ส่งออกรายงานผลล่าสุด""",
    language="python",
)
st.markdown("**อธิบาย:** ได้ไฟล์โมเดล `.pkl` สำหรับใช้งานต่อ และไฟล์ผลคาดการณ์ล่าสุดเพื่อนำไปแสดงผล/รายงาน")

st.header("3) ผลลัพธ์ที่ได้ (Outputs)")
st.markdown(
    """
- **dataset_financials_ttm.csv** — ชุดข้อมูล TTM ที่ใช้สร้างโมเดล  
- **feature_importance.csv** — ความสำคัญของแต่ละฟีเจอร์ (ใช้ตีความโมเดล)  
- **predictions_latest.csv** — ผลคาดการณ์ล่าสุดต่อหุ้น (Prob. น่าซื้อ + ป้ายกำกับ)
"""
)

st.header("4) การรันและ Deployment")
st.code("pip install -r requirements.txt", language="bash")
st.code("streamlit run app.py", language="bash")

st.header("5) สรุปการนำเสนอ (Conclusion Slide)")
st.markdown(
    """
- ✅ ดึงข้อมูลอัตโนมัติจาก **Yahoo Finance**  
- ✅ วิเคราะห์ **ปัจจัยพื้นฐาน 10 ตัว**  
- ✅ ใช้ **Random Forest** จำแนกว่า “หุ้นน่าซื้อ”  
- ✅ **Export** รายงานผลและความสำคัญของฟีเจอร์  
- 🚀 **ต่อยอดได้:** Dashboard ติดตามผล / ระบบแจ้งเตือน (Alert)
"""
)

st.caption("หมายเหตุ: หน้านี้จัดรูปแบบ ‘ข้อความล้วน’ ตามสไตล์เอกสารสรุป เพื่อใช้พรีเซนต์ได้ทันที")
