import os
import streamlit as st
from pathlib import Path

ASSET_DIR = Path("assets/ml_snippets_1")
SUPPORT_EXT = (".png", ".jpg", ".jpeg", ".webp")

st.set_page_config(page_title="ML Snippets #1", page_icon="🧠", layout="wide")

st.title("🧠 ML Snippets #1")
st.caption("รวมรูปโค้ด/โน้ตสรุป (อัปไฟล์ไว้ที่ `assets/ml_snippets_1/`)")

st.info("Tip: ตั้งชื่อไฟล์แบบมีเลขนำหน้า เช่น `01_`, `02_` เพื่อจัดลำดับการแสดงผลได้ง่ายขึ้น")

def list_images(folder: Path):
    if not folder.exists():
        return []
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORT_EXT]
    # natural sort by name
    imgs = sorted(imgs, key=lambda p: p.name.lower())
    return imgs

imgs = list_images(ASSET_DIR)

if not imgs:
    st.warning("ยังไม่พบรูปในโฟลเดอร์ `assets/ml_snippets_1/` — ลองอัปโหลดรูป PNG/JPG ลงโฟลเดอร์นี้ใน GitHub แล้วกด Deploy ใหม่")
else:
    # แสดงเป็น 2 คอลัมน์แบบ Responsive
    cols = st.columns(2)
    for i, p in enumerate(imgs):
        with cols[i % 2]:
            st.image(str(p), use_container_width=True, caption=p.name)
