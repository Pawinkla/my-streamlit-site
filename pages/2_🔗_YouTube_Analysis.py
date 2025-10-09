# pages/2_📈_YouTube_Analysis.py
# -------------------------------------------------------------
# YouTube Data Analysis (Replica of your Notion dashboard)
# - Upload CSV (UTF-8)
# - Filters on the left
# - KPI cards (total channels / subs / views / avg views per video)
# - Category summaries (Subscribers & Views)
# - Top 20 channels by subscribers
#
# หมายเหตุ:
# - รองรับชื่อคอลัมน์ที่ต่างกันได้ (สคริปต์พยายาม map ให้)
# - ถ้าข้อมูลไม่มีคอลัมน์ยอดวิวต่อวิดีโอ จะคำนวณจาก Views / Videos
# -------------------------------------------------------------

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="YouTube Data Analysis",
    page_icon="📈",
    layout="wide",
)

# ---------------------------
# Helpers
# ---------------------------
THOUSANDS = "{:,}".format

def format_int(x):
    try:
        return THOUSANDS(int(x))
    except Exception:
        return "-"

def coalesce_columns(df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    จับคู่/รีเนมคอลัมน์จากหลายชื่อที่เป็นไปได้ -> ชื่อมาตรฐาน
    targets = {
        "standard_name": ["candidate1", "candidate2", ...]
    }
    """
    cols = {c.lower().strip(): c for c in df.columns}
    for std, cands in targets.items():
        for c in cands:
            key = c.lower().strip()
            if key in cols:
                df.rename(columns={cols[key]: std}, inplace=True)
                break
    return df

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def ensure_int(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------
# Sidebar — Upload & Filters
# ---------------------------
with st.sidebar:
    st.header("อัปโหลด / เลือกข้อมูล")
    up = st.file_uploader("อัปโหลด CSV (UTF-8)", type=["csv"], help="Limit 200MB per file : CSV")

    st.markdown("---")
    st.header("กรองข้อมูล")

# ---------------------------
# Load data
# ---------------------------
if up is None:
    st.info("อัปโหลดไฟล์ CSV จาก youtubers.me (หรือไฟล์ที่มีคอลัมน์: ชื่อช่อง, ประเทศ, หมวดหมู่, ปีเริ่ม, Subscribers, Views, Videos)")
    st.stop()

# ลองอ่านเป็น UTF-8, หากไม่ได้ลอง latin-1
try:
    df = pd.read_csv(up, encoding="utf-8")
except Exception:
    up.seek(0)
    df = pd.read_csv(up, encoding="latin-1")

# ทำงานกับชื่อคอลัมน์แบบ lowercase เพื่อจับคู่
df.columns = [c.strip() for c in df.columns]

# map ชื่อคอลัมน์ให้เป็นมาตรฐาน
df = coalesce_columns(
    df,
    {
        "channel": ["channel", "title", "name", "yt_channel", "yt_name", "username", "channel_title"],
        "country": ["country", "nation"],
        "category": ["category", "topics"],
        "started": ["started", "joined", "created_year", "year"],
        "subscribers": ["subscribers", "subs", "subscriber_count"],
        "views": ["views", "total_views", "view_count"],
        "videos": ["videos", "video_count", "uploads"],
        # บางไฟล์มี avg_views ให้มาอยู่แล้ว
        "avg_views": ["avg_views", "average_views", "avg_views_per_video", "view_per_video"]
    }
)

# แปลงตัวเลข
df = ensure_numeric(df, ["subscribers", "views", "videos", "avg_views"])
df = ensure_int(df, ["started"])

# ถ้าไม่มี avg_views ให้คำนวณจาก views / videos
if "avg_views" not in df.columns:
    if "views" in df.columns and "videos" in df.columns:
        df["avg_views"] = df["views"] / df["videos"].replace(0, np.nan)
    else:
        df["avg_views"] = np.nan

# กรองค่าว่างของช่องสำคัญ
for must in ["channel", "category"]:
    if must in df.columns:
        df = df[~df[must].isna()]

# ---------------------------
# Sidebar filters
# ---------------------------
with st.sidebar:
    # Country
    if "country" in df.columns and df["country"].notna().any():
        countries = ["All"] + sorted([c for c in df["country"].dropna().unique().tolist() if str(c).strip() != ""])
        selected_country = st.selectbox("Country", countries, index=0)
    else:
        selected_country = "All"

    # Category
    if "category" in df.columns and df["category"].notna().any():
        cats = ["All"] + sorted([c for c in df["category"].dropna().unique().tolist() if str(c).strip() != ""])
        selected_cat = st.selectbox("Category", cats, index=0)
    else:
        selected_cat = "All"

    # Started years
    if "started" in df.columns and df["started"].notna().any():
        ymin, ymax = int(df["started"].min()), int(df["started"].max())
        year_range = st.slider("ปีที่เริ่มทำช่อง (Started)", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    else:
        year_range = None

    # Subscribers slider (ช่วง)
    if "subscribers" in df.columns and df["subscribers"].notna().any():
        smin, smax = int(df["subscribers"].min()), int(df["subscribers"].max())
        sub_range = st.slider("จำนวนผู้ติดตาม (Subscribers)", min_value=smin, max_value=smax, value=(smin, smax))
    else:
        sub_range = None

    # Search by channel name
    search_name = st.text_input("ค้นหาชื่อช่อง (ชื่อย่อ)", value="").strip().lower()

# ---------------------------
# Apply filters
# ---------------------------
f = df.copy()

if selected_country != "All" and "country" in f.columns:
    f = f[f["country"].astype(str) == selected_country]

if selected_cat != "All" and "category" in f.columns:
    f = f[f["category"].astype(str) == selected_cat]

if year_range and "started" in f.columns:
    ymin, ymax = year_range
    f = f[(f["started"] >= ymin) & (f["started"] <= ymax)]

if sub_range and "subscribers" in f.columns:
    smin, smax = sub_range
    f = f[(f["subscribers"] >= smin) & (f["subscribers"] <= smax)]

if search_name and "channel" in f.columns:
    f = f[f["channel"].astype(str).str.lower().str.contains(search_name)]

# ---------------------------
# Title + data source
# ---------------------------
st.title("YouTube Data Analysis")
st.caption("ที่มา (ตัวอย่าง): youtubers.me — วิเคราะห์ช่อง/วิดีโอยอดนิยม พร้อมกรองดู รายการ และดาวน์โหลดข้อมูล")

# ---------------------------
# KPI Cards
# ---------------------------
col1, col2, col3, col4 = st.columns(4)

# จำนวนช่อง
total_channels = len(f)

# Subscribers รวม
total_subs = f["subscribers"].sum() if "subscribers" in f.columns else np.nan

# Views รวม
total_views = f["views"].sum() if "views" in f.columns else np.nan

# วิวเฉลี่ย/วิดีโอ (รวม) = total_views / total_videos
if "views" in f.columns and "videos" in f.columns and f["videos"].sum() > 0:
    avg_per_video = int(f["views"].sum() / f["videos"].sum())
elif "avg_views" in f.columns:
    # ถ้าข้อมูลมี avg_views เป็นต่อช่อง เราใช้ค่าเฉลี่ย across channels แทน
    avg_per_video = int(f["avg_views"].mean(skipna=True)) if f["avg_views"].notna().any() else np.nan
else:
    avg_per_video = np.nan

col1.metric("จำนวนช่องทั้งหมด", format_int(total_channels))
col2.metric("รวมผู้ติดตาม (Subscribers)", format_int(total_subs))
col3.metric("รวมยอดวิว (Views)", format_int(total_views))
col4.metric("วิวเฉลี่ย/วิดีโอ", format_int(avg_per_video))

st.markdown("")

# ---------------------------
# Category Summary (Subscribers / Views)
# ---------------------------
left, right = st.columns(2)

if "category" in f.columns:
    # Subscribers by category
    if "subscribers" in f.columns and f["subscribers"].notna().any():
        cat_sub = (
            f.groupby("category", as_index=False)["subscribers"]
            .sum()
            .sort_values("subscribers", ascending=True)  # for horizontal ascending
        )
        fig1 = px.bar(
            cat_sub,
            x="subscribers",
            y="category",
            orientation="h",
            title="สรุปโดย Category – Subscribers รวม",
            labels={"subscribers": "Subscribers", "category": "Category"},
            height=500,
        )
        fig1.update_traces(marker_color="#2e77ff")
        fig1.update_layout(yaxis_title="", xaxis_title="")
        left.plotly_chart(fig1, use_container_width=True)
    else:
        left.info("ไม่มีคอลัมน์ Subscribers ให้สรุปตาม Category")

    # Views by category
    if "views" in f.columns and f["views"].notna().any():
        cat_view = (
            f.groupby("category", as_index=False)["views"]
            .sum()
            .sort_values("views", ascending=True)
        )
        fig2 = px.bar(
            cat_view,
            x="views",
            y="category",
            orientation="h",
            title="สรุปโดย Category – Views รวม",
            labels={"views": "Views", "category": "Category"},
            height=500,
        )
        fig2.update_traces(marker_color="#9aa3af")
        fig2.update_layout(yaxis_title="", xaxis_title="")
        right.plotly_chart(fig2, use_container_width=True)
    else:
        right.info("ไม่มีคอลัมน์ Views ให้สรุปตาม Category")
else:
    st.info("ไม่มีคอลัมน์ Category ในไฟล์ข้อมูล จึงไม่สามารถสรุปตาม Category ได้")

st.markdown("---")

# ---------------------------
# Top 20 Channels by Subscribers
# ---------------------------
st.subheader("🏆 Top 20 Channels (by Subscribers)")
if "channel" in f.columns and "subscribers" in f.columns:
    top20 = (
        f.dropna(subset=["channel", "subscribers"])
        .sort_values("subscribers", ascending=False)
        .head(20)
        .iloc[::-1]  # เพื่อให้แสดงจากมากไปน้อยในแนวนอน
    )
    fig_top = px.bar(
        top20,
        x="subscribers",
        y="channel",
        orientation="h",
        labels={"subscribers": "Subscribers", "channel": "Channel"},
        height=700,
    )
    fig_top.update_traces(marker_color="#f97316")
    fig_top.update_layout(yaxis_title="", xaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)
else:
    st.info("ต้องมีคอลัมน์ channel และ subscribers เพื่อแสดง Top 20 Channels")

# ---------------------------
# Optional: ตารางข้อมูลหลังกรอง (ถ้าต้องการเปิดดู)
# ---------------------------
with st.expander("ดูตารางข้อมูล (หลังกรอง)"):
    st.write(f.reset_index(drop=True))
