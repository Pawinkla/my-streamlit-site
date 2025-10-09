import os
import streamlit as st
from pathlib import Path

ASSET_DIR = Path("assets/ml_snippets_2")
SUPPORT_EXT = (".png", ".jpg", ".jpeg", ".webp")

st.set_page_config(page_title="ML Snippets #2", page_icon="📘", layout="wide")

st.title("📘 ML Snippets #2")
st.caption("รวมรูปโค้ด/โน้ตสรุป (อัปไฟล์ไว้ที่ `assets/ml_snippets_2/`)")

# optional: ค้นหา/กรองจากชื่อไฟล์
q = st.text_input("ค้นหาตามชื่อไฟล์ (optional)", "")

def list_images(folder: Path, keyword: str = ""):
    if not folder.exists():
        return []
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORT_EXT]
    imgs = sorted(imgs, key=lambda p: p.name.lower())
    if keyword:
        imgs = [p for p in imgs if keyword.lower() in p.name.lower()]
    return imgs

imgs = list_images(ASSET_DIR, q)

if not imgs:
    st.warning("ยังไม่พบรูปในโฟลเดอร์ `assets/ml_snippets_2/` หรือไม่ตรงกับคำค้นหา")
else:
    # แสดงเป็น 3 คอลัมน์ ถ้าหน้าจอกว้างจะดูแน่นขึ้น
    cols = st.columns(3)
    for i, p in enumerate(imgs):
        with cols[i % 3]:
            st.image(str(p), use_container_width=True, caption=p.name)
