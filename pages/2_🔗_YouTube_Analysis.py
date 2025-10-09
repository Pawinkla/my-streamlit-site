# pages/2_📈_YouTube_Analysis.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="YouTube Data Analysis", page_icon="📈", layout="wide")

NOTION_URL = "https://light-sock-d82.notion.site/Data-Scraping-YouTube-top-1000-vdo-9ba2c1ae8e7249bea86d2bffd5b05d70"

st.title("YouTube Data Analysis (Notion)")
st.caption("หน้านี้ฝังมาจาก Notion แบบเต็มหน้า ถ้าฝังไม่ขึ้น กดเปิดบน Notion แท็บใหม่")
st.link_button("เปิดบน Notion (แท็บใหม่)", NOTION_URL)

# ฝังหน้า Notion แบบเต็มความกว้าง
components.iframe(src=NOTION_URL, height=1200)
