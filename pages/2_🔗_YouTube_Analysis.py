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
st.image("../assets/top_youtube_thailand.jpg", use_column_width=True)



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
placeholder(260, "ตารางอันดับ (ภาพตัวอย่าง)")

section_divider()

# ---------- บล็อกโค้ด (ภาพ/โค้ดตัวอย่าง) ----------
st.markdown("#### ตัวอย่างโค้ด (import / requests / bs4 / pandas / matplotlib / seaborn)")
placeholder(280, "โค้ดตัวอย่าง (ภาพ)")

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
placeholder(200, "โค้ดตัวอย่าง: request + BeautifulSoup + ตาราง (ภาพ)")

st.markdown(
    """
- `response.encoding` : ตั้งค่า decoding (เช่น UTF-8)  
- `soup.find('table', class_='top-charts')` : เจาะตารางที่ต้องการ  
- `pd.DataFrame(...)` : แปลงข้อมูลเป็น DataFrame แล้วตั้งชื่อคอลัมน์  
- `display(df.head())` : แสดงตัวอย่างข้อมูล 5 แถว  
    """
)

# สคริปต์วนลูปอ่านแถว/คอลัมน์
placeholder(280, "โค้ดวนลูปอ่านแถว/คอลัมน์ (ภาพ)")

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
placeholder(300, "กราฟ matplotlib (ภาพ)")

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

section_divider()

# ---------- บทสรุปภาพรวม (ข้อ bullet) ----------
st.markdown(
    """
### สรุปแนวทางการทำงาน
- โหลดเว็บเพจด้วย `requests`  
- ใช้ `BeautifulSoup` แปลง/ค้นหา HTML (ตาราง/คอลัมน์)  
- ใช้ `pandas.read_html()` หรือสร้าง DataFrame เอง  
- ลบช่องว่าง/ปรับชื่อคอลัมน์เป็น snake case  
- แสดงตัวอย่างข้อมูลด้วย `df.head()`  
    """,
)

placeholder(260, "โค้ดตัวอย่าง (ภาพ)")

# ---------- ส่วน “จัดการข้อความไทย/ฟอนต์/โมจิบาเกะ” ----------
section_divider()
st.markdown("### การตั้งค่าฟอนต์และแก้ encoding (mojibake) สำหรับชื่อช่องภาษาไทย")
placeholder(340, "โค้ด: ติดตั้ง/ตั้งค่าฟอนต์ + ฟังก์ชันแก้ mojibake (ภาพ)")

st.markdown(
    """
- ติดตั้งและตั้งค่าฟอนต์ภาษาไทย (เช่น Loma / Noto Sans Thai) เพื่อให้กราฟ/ข้อความไทยแสดงถูกต้อง  
- เขียนฟังก์ชันช่วยแก้โมจิบาเกะ (mojibake) เช่น บางตัวแสดงเป็น `\\u0E00`–`\\u0E7F`  
- แปลง encoding (เช่นจาก `'latin1'` → `'utf-8'`) เมื่อจำเป็น แล้วใช้ `.apply()` ช่วยปรับใน DataFrame  
    """
)

placeholder(240, "รูปโค้ด: ฟังก์ชัน fix_mojibake + การใช้งานกับ DataFrame (ภาพ)")

st.info(
    "หากข้อมูลเป็นไทยถูกต้องอยู่แล้ว (ตรวจเจอช่วง `\\u0E00`–`\\u0E7F`) ไม่ต้องแก้เพิ่ม — "
    "หากไม่ใช่จึงค่อยลองแปลง encoding เป็น `utf-8` และปรับชื่อคอลัมน์/ตัวอักษรให้เหมาะสม"
)

# ลิ้งก์ภายนอก (ตัวอย่าง)
st.caption(
    'ลิงก์ประกอบ (ตัวอย่าง): '
    '<a href="https://colab.research.google.com/drive/1A_KrdsoAuIth24gA3igPFsUzqYtde33s?usp=sharing" target="_blank">'
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
