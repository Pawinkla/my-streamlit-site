
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="YouTube Analysis", page_icon="📈")

NOTION_URL = "https://www.notion.so/Data-Scraping-YouTube-top-1000-vdo-9ba2c1ae8e7249bea86d2bffd5b05d70?source=copy_link"

tab1, tab2 = st.tabs(["🔗 เปิด Notion โดยตรง", "🧩 ฝัง Notion ในหน้านี้"])

with tab1:
    st.link_button("เปิดใน Notion", NOTION_URL)

with tab2:
    st.caption("ถ้าไม่แสดง ให้เปิด Share → Publish to web บน Notion ก่อน")
    components.iframe(src=NOTION_URL, height=900)   # ลอง 900–1200 ถ้าอยากสูงขึ้น
