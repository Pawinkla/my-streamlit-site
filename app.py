import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Pawin • Profile",
    page_icon="👤",
    layout="wide"
)

# ---------- Minimal CSS theming ----------
st.markdown("""
<style>
/* page padding */
.block-container {padding-top: 2.2rem; padding-bottom: 2rem;}

/* header */
.hero {
  background: linear-gradient(90deg, #0ea5e9 0%, #22c55e 50%, #a855f7 100%);
  border-radius: 18px;
  padding: 28px;
  color: white;
  box-shadow: 0 10px 26px rgba(2,8,23,.20);
}
.hero h1{ margin: 0 0 6px 0; font-weight: 800; letter-spacing:.2px;}
.hero .sub{opacity:.95; margin-top:4px;}

/* avatar */
.avatar {
  border-radius: 18px;
  width: 140px; height: 140px; object-fit: cover;
  box-shadow: 0 12px 24px rgba(2,8,23,.18);
  border: 4px solid rgba(255,255,255,.85);
}

/* chip / badge */
.chip {
  display:inline-block; padding:6px 10px; margin: 4px 6px 0 0;
  border-radius: 14px; background:#f1f5f9; border:1px solid #e2e8f0;
  font-size:.88rem; color:#0f172a;
}

/* card */
.card {
  border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px 18px;
  background: white; box-shadow: 0 6px 18px rgba(2,8,23,.06);
}
.card h3{margin: 0 0 6px 0}
.card p{margin:.25rem 0 .5rem 0; color:#334155}
.card .tags span{
  display:inline-block; background:#eef2ff; color:#3730a3;
  font-size:.78rem; padding:4px 8px; border-radius: 10px; margin-right:6px;
}

/* subtle link buttons */
.link-btn{
  display:inline-block; padding:8px 12px; border-radius:12px;
  border:1px solid #e2e8f0; text-decoration:none; font-weight:600;
  transition:all .15s ease; margin-right:8px; color:#0f172a;
}
.link-btn:hover{ background:#0ea5e9; color:#fff; border-color:#0ea5e9; }

/* foot note */
.note {color:#64748b; font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
with st.container():
    colA, colB = st.columns([1, 5], vertical_alignment="center")
    with colA:
        st.image("assets/Pawin.jpg", use_column_width=False, caption=None, output_format="auto")
    with colB:
        st.markdown("""
<div class="hero">
  <h1>ภวินท์ กล้าทำ</h1>
  <div class="sub">Data Science • Machine Learning • Streamlit</div>
  <div style="margin-top:12px;">
    <a class="link-btn" href="mailto:someone@example.com" target="_blank">✉️ ติดต่อ</a>
    <a class="link-btn" href="https://github.com/" target="_blank">🐙 GitHub</a>
    <a class="link-btn" href="https://www.linkedin.com/" target="_blank">🔗 LinkedIn</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ===================== SUMMARY METRICS =====================
m1, m2, m3 = st.columns(3)
with m1: st.metric("Projects", "3+", help="จำนวนงานหลักที่โชว์ด้านล่าง")
with m2: st.metric("Focus", "ML / DS", help="สายงานที่ถนัดและสนใจ")
with m3: st.metric("Stack", "Python-first", help="Python, SQL, PyTorch, Streamlit")

# ===================== ABOUT & SKILLS =====================
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.subheader("เกี่ยวกับฉัน")
    st.write(
        "นักศึกษาปี 4 สาย Data Science / Machine Learning "
        "ชอบทำโปรเจกต์ที่จับต้องได้และนำเสนอผ่าน Streamlit ให้ใช้งานง่าย • "
        "สนใจงานด้านข้อมูล ตั้งแต่ดึงข้อมูล-แปลงข้อมูล-ฝึกโมเดล-และเล่าเรื่องผ่านภาพ"
    )

    st.subheader("ทักษะ (Tech Stack)")
    st.markdown(
        """
        <span class="chip">Python</span>
        <span class="chip">Pandas</span>
        <span class="chip">SQL</span>
        <span class="chip">scikit-learn</span>
        <span class="chip">PyTorch</span>
        <span class="chip">Streamlit</span>
        <span class="chip">Visualization</span>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.subheader("ข้อมูลติดต่อ")
    st.write("📧 อีเมล: someone@example.com")
    st.write("🌐 เว็บไซต์/Portfolio: https://example.com")
    st.write("🏠 ที่อยู่: Bangkok, TH")

# ===================== PROJECTS =====================
st.markdown("### โปรเจกต์เด่น")

p1, p2, p3 = st.columns(3, gap="large")

with p1:
    st.markdown(
        """
        <div class="card">
          <h3>🎬 YouTube Data Analysis</h3>
          <p>ดึง-สรุป-และแสดงผลวิดีโอยอดนิยม พร้อมสคริปต์อธิบายการทำงาน</p>
          <div class="tags">
            <span>Pandas</span><span>BeautifulSoup</span><span>Visualization</span>
          </div>
          <div style="margin-top:10px;">
        """,
        unsafe_allow_html=True,
    )
    # ลิงก์ไปเพจย่อย (ใช้ได้บน Streamlit >= 1.31)
    try:
        st.page_link("pages/2_📈_YouTube_Analysis.py", label="เปิดโปรเจกต์", use_container_width=True)
    except Exception:
        st.link_button("เปิดโปรเจกต์", "#", use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

with p2:
    st.markdown(
        """
        <div class="card">
          <h3>📊 Stock Scanner (Fundamental ML)</h3>
          <p>สแกนหุ้นด้วยตัวชี้วัดพื้นฐาน + โมเดล ML เพื่อคัดกรองเบื้องต้น</p>
          <div class="tags">
            <span>scikit-learn</span><span>Pandas</span><span>Finance</span>
          </div>
          <div style="margin-top:10px;">
        """,
        unsafe_allow_html=True,
    )
    try:
        st.page_link("pages/3_📈_Stock_ML.py", label="เปิดโปรเจกต์", use_container_width=True)
    except Exception:
        st.link_button("เปิดโปรเจกต์", "#", use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

with p3:
    st.markdown(
        """
        <div class="card">
          <h3>🥗 Healthy vs Junk Food</h3>
          <p>Image Classification (ResNet18) บน Streamlit พร้อมรายงานความมั่นใจ</p>
          <div class="tags">
            <span>PyTorch</span><span>ResNet18</span><span>Streamlit</span>
          </div>
          <div style="margin-top:10px;">
        """,
        unsafe_allow_html=True,
    )
    try:
        st.page_link("pages/4_🥗_Healthy_vs_Junk_Food.py", label="เปิดโปรเจกต์", use_container_width=True)
    except Exception:
        st.link_button("เปิดโปรเจกต์", "#", use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ===================== FOOT NOTE =====================
st.markdown(
    '<div class="note">Tip: ใช้เมนูซ้ายมือเพื่อสลับหน้า หรือกดปุ่ม “เปิดโปรเจกต์” ในการ์ดด้านบน</div>',
    unsafe_allow_html=True
)
