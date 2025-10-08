
import streamlit as st
st.set_page_config(page_title="Profile", page_icon="👤", layout="wide")
st.title("👤 Profile Page")

left, right = st.columns([1, 2])
with left:
    st.image("assets/Pawin.jpg", width=160)  # เปลี่ยนเป็นรูปของคุณ
with right:
    st.markdown("""
**ชื่อ:ภวินท์ กล้าทำ**  
**สาขา/ชั้นปี:4**   
**ความสนใจ:** Data Science, Machine Learning, Streamlit  
**ทักษะ:** Python, SQL, scikit-learn, PyTorch, Streamlit, Visualization
""")

st.subheader("Projects")
st.markdown("""
- **YouTube Data Analysis** — วิเคราะห์วิดีโอยอดนิยม (ลิงก์/ฝัง Notion)  
- **Stock Scanner (Fundamental ML)** — สแกนหุ้นจากปัจจัยพื้นฐาน  
- **Healthy vs Junk Food** — Image Classification ด้วย ResNet18 + Streamlit  
""")

st.info("เลือกหน้าอื่น ๆ จากเมนูซ้ายมือได้เลย")
