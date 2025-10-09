# pages/2_📺_YouTube_Analysis.py
import streamlit as st

st.set_page_config(
    page_title="YouTube Data Analysis",
    page_icon="📺",
    layout="wide",
)

# ---------- Utils ----------
def placeholder(height: int = 260, label: str = ""):
    """
    กล่องสีขาวสำหรับแทนรูป/กราฟ/โค้ด
    """
    st.markdown(
        f"""
        <div style="
            width:100%;
            height:{height}px;
            background:#ffffff;
            border:1px solid #e6e6e6;
            border-radius:8px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#888;
            font-size:14px;
        ">{label if label else "placeholder"}</div>
        """,
        unsafe_allow_html=True,
    )

def section_divider():
    st.markdown(
        """<hr style="margin:2.2rem 0; border:none; border-top:1px solid #eee;">""",
        unsafe_allow_html=True,
    )

# ---------- Banner (แดงสไตล์ YouTube) ----------
st.markdown(
    """
    <div style="
        width:100%;
        height:220px;
        background:linear-gradient(180deg, #E41010, #B80707);
        border-radius:12px;
        margin-bottom:18px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#fff;
        font-size:48px;
        font-weight:800;
        letter-spacing:1px;
    ">
      YOUTUBE
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Title ----------
st.markdown(
    """
    <h1 style="margin:0 0 4px 0; font-weight:800;">
      Data Scraping - YouTube top 1000 vdo
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="margin:4px 0 22px 0;">ชื่อเว็บ : <a href="https://youtubers.me" target="_blank">Youtubers.me</a></div>',
    unsafe_allow_html=True,
)

# ภาพตารางอันดับ (แทนด้วย placeholder)
st.image("assets/top_youtube_thailand.jpg", use_column_width=True)

section_divider()

# ---------- ส่วน Assignment ----------
st.markdown(
    """
    <h2 style="margin:0 0 12px 0;">Assignment #4</h2>
    <div style="opacity:.9;">
      <b>หัวข้อ:</b> Data Scraping – YouTube Top 1000 Videos และเขียนสรุปเนื้อหาที่น่าสนใจที่เว็บไซต์ notion.so
    </div>
    """,
    unsafe_allow_html=True,
)

# โค้ดกล่องภาพตาราง (ย่อ) 
st.caption("รูปตัวอย่างตารางอันดับ (ย่อ)")
st.image("assets/code1.jpg", use_column_width=True)

section_divider()

# ---------- บล็อกโค้ด (ภาพ/โค้ดตัวอย่าง) ----------
st.markdown("#### ตัวอย่างโค้ด (import / requests / bs4 / pandas / matplotlib / seaborn)")

# สรุป bullet list
st.markdown(
    """
- `requests` : โหลดข้อมูลจากเว็บ  
- `bs4 (BeautifulSoup)` : แยกข้อมูลจากหน้าเว็บ  
- `re` : หา/จัดการข้อมูลแบบ (regex)  
- `pandas` : จัดการข้อมูลตาราง  
- `datetime` : จัดการวัน-เวลา  
- `seaborn / matplotlib` : วาดกราฟแสดงผล
    """
)

# บล็อกภาพ/โค้ดเพิ่มเติม
st.image("assets/code2.jpg", use_column_width=True)

st.markdown(
    """
- `response.encoding` : ตั้งค่า decoding (เช่น UTF-8)  
- `soup.find('table', class_='top-charts')` : เจาะตารางที่ต้องการ  
- `pd.DataFrame(...)` : แปลงข้อมูลเป็น DataFrame แล้วตั้งชื่อคอลัมน์  
- `display(df.head())` : แสดงตัวอย่างข้อมูล 5 แถว  
    """
)

# สคริปต์วนลูปอ่านแถว/คอลัมน์
st.image("assets/code3.jpg", use_column_width=True)

st.markdown(
    """
**เก็บข้อมูลแต่ละคอลัมน์ เช่น**
- อันดับ (`rank`)  
- ชื่อวิดีโอ (`video_name`)  
- ชื่อช่อง (`channel_name`)  
- ยอดดู (`vdo_views`)  
- ไลค์ (`likes`)  
- ไม่ชอบ (`dislikes`)  
- หมวดหมู่ (`category`)  
- วันที่โพสต์ (`published`)  
    """
)

# กราฟ matplotlib (ภาพ)
st.image("assets/code4.jpg", use_column_width=True)

st.markdown(
    """
- `plt.figure(figsize=(12, 6))` : ตั้งขนาดกราฟ  
- `plt.bar(...)` : วาดกราฟแท่งตามหมวดหมู่ (เช่นจาก `vdo_views`)  
- `plt.xticks(rotation=45, ha='right')` : ให้ชื่ออ่านง่าย  
- `plt.xlabel() / plt.ylabel() / plt.title()` : กำหนดข้อความกำกับ  
- `plt.tight_layout()` : จัดเลย์เอาต์ไม่ให้ทับกัน  
- `plt.show()` : แสดงกราฟ  
    """
)
st.image("assets/code5.jpg", use_column_width=True)
section_divider()

st.markdown(
    """
- `โหลดเว็บเพจด้วย requests  
- `ใช้ BeautifulSoup แปลง HTML ของเว็บ  
- `หา <table> แรกในหน้าเว็บ  
- `ใช้ pandas.read_html() แปลงตาราง HTML เป็น DataFrame
- `ลบช่องว่างชื่อคอลัมน์
- `เปลี่ยนชื่อคอลัมน์บางอันให้สั้นลง เช่น 'Video title' → 'Vdo_name', 'Likes' → 'likes'
- `แสดงข้อมูล 5 แถวแรก
    """
)
st.image("assets/code6.jpg", use_column_width=True)
st.image("assets/code7.jpg", use_column_width=True)
st.image("assets/code10.jpg", use_column_width=True)

section_divider()

st.markdown(
    """
- ติดตั้งและตั้งค่าฟอนต์ภาษาไทย (Loma) เพื่อให้ matplotlib แสดงผลข้อความภาษาไทยได้ถูกต้อง  
- ดึงข้อมูลตารางจากเว็บ โดยใช้ requests และ BeautifulSoup
- แปลงข้อมูลในตารางเป็น DataFrame ด้วย pandas
- ทำความสะอาดข้อมูลในคอลัมน์ likes ให้เป็นตัวเลขจริง
- แก้ไขปัญหา encoding ของชื่อช่องที่อาจแสดงเป็นตัวอักษรเพี้ยน (mojibake) ด้วยฟังก์ชัน fix_double_encoded
- รวมยอดไลก์ตามชื่อช่อง และกรองเฉพาะช่องที่ยอดไลก์เกิน 1 ล้าน
- ตัดชื่อช่องให้สั้นลงสำหรับแสดงในกราฟ
- วาดกราฟแท่งแสดงจำนวนไลก์รวมของ 5 ช่องที่มีไลก์สูงสุด โดยใช้ฟอนต์ไทยที่ตั้งไว้ เพื่อให้ชื่อช่องและข้อความแสดงผลเป็นภาษาไทยอย่างถูกต้อง
    """
)

st.image("assets/code11.jpg", use_column_width=True)
st.markdown(
    """
- fix_mojibake เป็นฟังก์ชันแก้ข้อความที่ encoding ผิด (mojibake)
- ใช้การเข้ารหัสใหม่จาก 'latin1' แล้วถอดรหัสเป็น 'utf-8' เพื่อคืนค่าข้อความที่ถูกต้อง 
- ใช้ .apply() กับคอลัมน์ชื่อช่อง เพื่อแก้ปัญหาข้อความเพี้ยนในช่องชื่อของ DataFrame 
    """
)
st.image("assets/code12.jpg", use_column_width=True)

st.markdown(
    """
- ถ้าในข้อความนั้นมีตัวอักษรภาษาไทยอยู่แล้ว (\u0E00 ถึง \u0E7F) ถือว่าข้อความถูกต้องแล้ว ไม่ต้องแก้ไข
- แต่ถ้าไม่มีตัวอักษรไทยเลย ให้ลองแปลง encoding จาก 'latin1' เป็น 'utf-8'
- ถ้าหลังแปลง encoding แล้วยังเจอตัวอักษรไทยในข้อความ จะคืนค่าข้อความที่แก้ไขแล้ว (fixed)
- ถ้าแปลงไม่ได้หรือไม่มีตัวอักษรไทยในข้อความหลังแปลง ก็คืนค่าเดิม
    """
)
# ลิ้งก์ภายนอก (ตัวอย่าง)
st.caption(
    'ลิงก์ประกอบ : '
    '<a href="https://colab.research.google.com/drive/1A_KrdsoAuIth24gA3igFPsUzqYtde33s?usp=sharing" target="_blank">'
    'Google Colab</a>',
    unsafe_allow_html=True,
)

section_divider()
st.markdown(
    """
    <div style="opacity:.8;">
      หน้านี้จัดทำใหม่บน Streamlit เพื่อให้ใช้งานได้โดยไม่ต้องพึ่ง Notion — 
      จุดที่เป็นภาพ/กราฟ/โค้ด ได้วางกล่องสีขาวไว้ให้แทนก่อนตามตำแหน่งเดิม สามารถค่อย ๆ แทนที่ด้วยรูปจริง/กราฟจริงได้ภายหลัง
    </div>
    """,
    unsafe_allow_html=True,
)
