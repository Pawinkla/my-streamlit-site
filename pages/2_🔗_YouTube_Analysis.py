# pages/2_📈_YouTube_Analysis.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime

# ---------------------------
# Page config & banner
# ---------------------------
st.set_page_config(page_title="YouTube Data Analysis", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .big-title {font-size: 36px; font-weight: 800; margin-bottom: 0;}
    .subtle {color:#6b7280}
    .card {background:#fff; border:1px solid #eee; border-radius:14px; padding:18px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">YouTube Data Analysis</div>', unsafe_allow_html=True)
st.caption("ที่มา (ตัวอย่าง): youtubers.me — วิเคราะห์ช่อง/วิดีโอยอดนิยม พร้อมกรอง ดูกราฟ และดาวน์โหลดข้อมูล")

# ---------------------------
# Data loader (CSV / Upload / Demo)
# ---------------------------
@st.cache_data
def load_csv_from_disk():
    # วางไฟล์เองที่ data/youtube_top_th.csv (columns ที่คาดหวังดูฟังก์ชัน normalize_columns ด้านล่าง)
    try:
        return pd.read_csv("data/youtube_top_th.csv")
    except Exception:
        return None

def demo_data(n=200):
    rng = np.random.default_rng(42)
    cats = ["Entertainment","Music","Education","News","Gaming","Sports","Science & Tech","People & Blogs"]
    start_years = rng.integers(2010, 2024, size=n)
    df = pd.DataFrame({
        "channel": [f"ช่อง {i+1}" for i in range(n)],
        "subscribers": rng.integers(50_000, 15_000_000, size=n),
        "view_count": rng.integers(1_000_000, 4_000_000_000, size=n),
        "video_count": rng.integers(20, 4000, size=n),
        "category": rng.choice(cats, size=n),
        "started": start_years,
        "country": ["Thailand"]*n,
        "url": [f"https://youtube.com/@demo{i}" for i in range(n)],
    })
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    พยายาม map คอลัมน์ที่พบบ่อย → คอลัมน์มาตรฐานที่หน้าแอปใช้
    รองรับหัวข้อที่ต่างกันบางส่วน เช่น channel_name → channel
    """
    rename_map = {
        "channel_name": "channel",
        "channelTitle": "channel",
        "title": "channel",

        "subs": "subscribers",
        "subscriberCount": "subscribers",

        "views": "view_count",
        "viewCount": "view_count",

        "videos": "video_count",
        "videoCount": "video_count",

        "categoryName": "category",
        "genre": "category",

        "year_started": "started",
        "started_year": "started",

        "countryCode": "country",
        "link": "url",
        "channel_url": "url",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    # เฉพาะคอลัมน์ที่ต้องใช้
    keep = ["channel","subscribers","view_count","video_count","category","started","country","url"]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    # แปลงชนิด
    for col in ["subscribers","view_count","video_count","started"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["category"] = df["category"].astype(str)
    df["country"]  = df["country"].astype(str)
    return df[keep]

st.sidebar.header("📥 อัปโหลด / เลือกข้อมูล")
uploaded = st.sidebar.file_uploader("อัปโหลด CSV (UTF-8)", type=["csv"])
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    data = normalize_columns(raw)
else:
    disk = load_csv_from_disk()
    if disk is not None:
        data = normalize_columns(disk)
    else:
        data = normalize_columns(demo_data())

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🔎 กรองข้อมูล")
countries = sorted([c for c in data["country"].dropna().unique() if c and c != "nan"])
country = st.sidebar.selectbox("Country", options=["All"] + countries, index= (["All"]+countries).index("All"))

categories = sorted([c for c in data["category"].dropna().unique() if c and c != "nan"])
cat_sel = st.sidebar.multiselect("Category", options=categories, default=[])

min_year, max_year = int(np.nanmin(data["started"])) if data["started"].notna().any() else 2010, \
                     int(np.nanmax(data["started"])) if data["started"].notna().any() else datetime.now().year
year_range = st.sidebar.slider("ปีที่เริ่มทำช่อง (Started)", min_value=min_year, max_value=max_year,
                               value=(min_year, max_year), step=1)

min_subs = int(np.nanmin(data["subscribers"])) if data["subscribers"].notna().any() else 0
max_subs = int(np.nanmax(data["subscribers"])) if data["subscribers"].notna().any() else 10_000_000
sub_range = st.sidebar.slider("จำนวนผู้ติดตาม (Subscribers)", min_value=min_subs, max_value=max_subs,
                              value=(min_subs, max_subs), step=max(1, (max_subs-min_subs)//100))

search_kw = st.sidebar.text_input("ค้นหาช่อง (ชื่อช่อง)")

# ---------------------------
# Apply filters
# ---------------------------
df = data.copy()
if country != "All":
    df = df[df["country"].fillna("").str.contains(country, case=False)]
if cat_sel:
    df = df[df["category"].isin(cat_sel)]
df = df[df["started"].between(year_range[0], year_range[1], inclusive="both")]
df = df[df["subscribers"].between(sub_range[0], sub_range[1], inclusive="both")]
if search_kw.strip():
    kw = search_kw.strip().lower()
    df = df[df["channel"].fillna("").str.lower().str.contains(kw)]

# ---------------------------
# KPI Cards
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("จำนวนช่อง", f"{len(df):,}")
c2.metric("รวมผู้ติดตาม", f"{int(np.nansum(df['subscribers'])):,}")
c3.metric("รวมยอดวิว", f"{int(np.nansum(df['view_count'])):,}")
avg_views = np.nanmean(df['view_count']/df['video_count']) if df['video_count'].gt(0).any() else np.nan
c4.metric("วิวเฉลี่ย/วิดีโอ", f"{0 if np.isnan(avg_views) else int(avg_views):,}")

st.divider()

# ---------------------------
# Charts
# ---------------------------
st.subheader("📊 สรุปโดย Category")
top_cat = (
    df.groupby("category", dropna=True)
      .agg(subscribers=("subscribers","sum"), views=("view_count","sum"), channels=("channel","count"))
      .reset_index()
      .sort_values("subscribers", ascending=False)
      .head(15)
)

col_a, col_b = st.columns(2)
with col_a:
    if not top_cat.empty:
        st.altair_chart(
            alt.Chart(top_cat).mark_bar().encode(
                x=alt.X("subscribers:Q", title="ผู้ติดตามรวม"),
                y=alt.Y("category:N", sort="-x", title=None),
                tooltip=["category","channels","subscribers","views"]
            ).properties(height=420),
            use_container_width=True
        )
    else:
        st.info("ไม่มีข้อมูลในช่วงการกรอง")

with col_b:
    if not top_cat.empty:
        st.altair_chart(
            alt.Chart(top_cat).mark_bar(color="#94a3b8").encode(
                x=alt.X("views:Q", title="ยอดวิวรวม"),
                y=alt.Y("category:N", sort="-x", title=None),
                tooltip=["category","channels","subscribers","views"]
            ).properties(height=420),
            use_container_width=True
        )
    else:
        st.info("ไม่มีข้อมูลในช่วงการกรอง")

st.subheader("🏆 Top 20 Channels (by Subscribers)")
top_ch = df.sort_values("subscribers", ascending=False).head(20)
if not top_ch.empty:
    st.altair_chart(
        alt.Chart(top_ch).mark_bar().encode(
            x=alt.X("subscribers:Q", title="ผู้ติดตาม"),
            y=alt.Y("channel:N", sort="-x", title=None),
            color=alt.Color("category:N", legend=None),
            tooltip=["channel","category","subscribers","view_count","video_count","started"]
        ).properties(height=520),
        use_container_width=True
    )
else:
    st.info("ไม่มีข้อมูลในช่วงการกรอง")

st.subheader("📈 Subscribers vs Views (bubble = วิดีโอ)")
if not df.empty:
    bubble = alt.Chart(df.dropna(subset=["subscribers","view_count","video_count"])).mark_circle(opacity=0.6).encode(
        x=alt.X("subscribers:Q", title="Subscribers", scale=alt.Scale(type="log")),
        y=alt.Y("view_count:Q", title="Views", scale=alt.Scale(type="log")),
        size=alt.Size("video_count:Q", title="Videos", legend=None),
        color=alt.Color("category:N", legend=alt.Legend(title="Category")),
        tooltip=["channel","category","subscribers","view_count","video_count","started"]
    ).properties(height=520)
    st.altair_chart(bubble, use_container_width=True)
else:
    st.info("ไม่มีข้อมูลในช่วงการกรอง")

st.divider()

# ---------------------------
# Data Table + Download
# ---------------------------
st.subheader("ตารางข้อมูล (หลังกรอง)")
show_cols = ["channel","category","subscribers","view_count","video_count","started","country","url"]
st.dataframe(df[show_cols].sort_values("subscribers", ascending=False), use_container_width=True, height=420)

csv = df[show_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ ดาวน์โหลดข้อมูล (CSV, UTF-8)", csv, file_name="youtube_filtered.csv", mime="text/csv")

# ข้อความช่วยเหลือด้านล่าง
with st.expander("ℹ️ วิธีเตรียมไฟล์ CSV ให้ใช้งานได้ดี"):
    st.markdown(
        """
        ใช้คอลัมน์มาตรฐาน (หรือ map อัตโนมัติ):
        - **channel**, **subscribers**, **view_count**, **video_count**, **category**, **started**, **country**, **url**  
        ถ้าไฟล์ของคุณใช้ชื่ออื่น (เช่น `channel_name`, `subs`, `views`, `videos`, `year_started` ฯลฯ)  
        แอปจะพยายามแปลงให้โดยอัตโนมัติ
        """
    )
