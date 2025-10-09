# -*- coding: utf-8 -*-
# Streamlit page: โชว์โค้ดโปรเจกต์ Healthy vs Junk Food (4 หัวข้อใหญ่ + ต่อยอด)
# วางไฟล์นี้ไว้ที่ pages/6_📘_Show_Code_Food_Project.py

import streamlit as st
from pathlib import Path
import re
import textwrap

st.set_page_config(page_title="โชว์โค้ด Healthy vs Junk Food", page_icon="📘", layout="wide")
st.title("📘 โชว์โค้ดโปรเจกต์: Healthy vs Junk Food")
st.caption("สรุป 4 ส่วนหลักของงาน + โค้ดสำคัญที่ใช้จริง พร้อมแนวทางต่อยอด")

ROOT = Path(".")
FILES = {
    "mapping": ROOT / "mapping_rules.py",
    "prepare": ROOT / "prepare_dataset.py",
    "train":   ROOT / "train.py",
    "predict": ROOT / "predict.py",
    "readme":  ROOT / "README.md",
}

def read_text(p: Path) -> str:
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return p.read_text(errors="ignore")
    return f"# ไม่พบไฟล์: {p}"

def cut_block(text: str, pattern: str, fallback_all: bool = True) -> str:
    """
    ดึงเฉพาะบล็อกโค้ดที่สนใจด้วย regex pattern (DOTALL).
    ถ้าไม่เจอและ fallback_all=True จะคืนทั้งไฟล์
    """
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        return textwrap.dedent(m.group(0)).strip()
    return text if fallback_all else "# (ไม่พบบล็อกที่ต้องการ)"

# --------------------------
# 1) Mapping Rules
# --------------------------
st.header("1) กฎจัดกลุ่มอาหาร (Rule-based Mapping)")
st.markdown(
    "- จุดประสงค์: แปลงชื่อโฟลเดอร์/คลาสดิบ → **Healthy / Unhealthy**\n"
    "- แนวคิด: เช็กลิสต์คำที่เป็นสัญญาณของอาหารมัน/ทอด/หวานฯลฯ เทียบกับลิสต์อาหารคลีน/นึ่ง/ย่าง"
)

mapping_src = read_text(FILES["mapping"])
mapping_key = cut_block(mapping_src, r"def\s+map_class_to_label\(.*?^\s*return\s+\"Healthy\"\s*$")
with st.expander("ดูโค้ดสำคัญ: map_class_to_label()", expanded=True):
    st.code(mapping_key, language="python")
with st.expander("ดูไฟล์ mapping_rules.py ทั้งไฟล์"):
    st.code(mapping_src, language="python")

# --------------------------
# 2) เตรียมชุดข้อมูล (CSV)
# --------------------------
st.header("2) เตรียมข้อมูล Train/Val เป็น CSV")
st.markdown(
    "- สแกนโฟลเดอร์ `images/` → สร้าง `data/train.csv` และ `data/val.csv`\n"
    "- ทำ **stratified split** โดยคงสัดส่วนคลาสไม่ให้เอียง"
)

prep_src = read_text(FILES["prepare"])
prep_blocks = cut_block(prep_src, r"def\s+scan_images\(.*?def\s+write_csv\(.*?^if\s+__name__\s*==\s*\"__main__\":.*?$")
with st.expander("ดูโค้ดสำคัญ: scan_images / stratified_split / write_csv / main()", expanded=True):
    st.code(prep_blocks, language="python")
with st.expander("ดูไฟล์ prepare_dataset.py ทั้งไฟล์"):
    st.code(prep_src, language="python")

# --------------------------
# 3) เทรนโมเดล ResNet-18
# --------------------------
st.header("3) เทรนโมเดล ResNet-18 (Fine-tune)")
st.markdown(
    "- โหลดเวท ResNet-18 จาก ImageNet → ปรับชั้นสุดท้ายให้เหลือ 2 คลาส\n"
    "- ใช้ **CrossEntropyLoss + Adam**\n"
    "- เซฟ **best_model.pt** เมื่อ val accuracy ดีขึ้น"
)

train_src = read_text(FILES["train"])
train_key = cut_block(
    train_src,
    r"(weights\s*=\s*ResNet18_Weights\.IMAGENET1K_V1.*?model\.to\(device\).*?optimizer\s*=\s*torch\.optim\.Adam.*?for\s+epoch.*?print\(\"Best val acc\",\s*best_acc\))",
)
with st.expander("ดูโค้ดสำคัญ: สร้างโมเดล/ลูปเทรน/บันทึก best_model.pt", expanded=True):
    st.code(train_key, language="python")
with st.expander("ดูไฟล์ train.py ทั้งไฟล์"):
    st.code(train_src, language="python")

# --------------------------
# 4) ทำนายภาพใหม่ (Inference)
# --------------------------
st.header("4) นำโมเดลมาใช้งาน (Predict)")
st.markdown(
    "- โหลด `best_model.pt` → เตรียมรูปขนาด 224×224 → ทำนายความน่าจะเป็นสองคลาส\n"
    "- ใช้ใน CLI หรือฝังในแอปได้"
)

pred_src = read_text(FILES["predict"])
pred_key = cut_block(pred_src, r"(def\s+load_model\(.*?return\s+model,.*?mean,.*?std.*?def\s+main\(\):.*?if\s+__name__\s*==\s*\"__main__\":\s*main\(\))")
with st.expander("ดูโค้ดสำคัญ: โหลดโมเดล + พรีโปรเซส + ทำนาย", expanded=True):
    st.code(pred_key, language="python")
with st.expander("ดูไฟล์ predict.py ทั้งไฟล์"):
    st.code(pred_src, language="python")

# --------------------------
# คำสั่งที่ใช้จริง (ช่วยอธิบายบนเวที)
# --------------------------
st.subheader("คำสั่งที่รันจริง (สรุปสั้น ๆ)")
st.code(
    """# 1) เตรียม CSV
python prepare_dataset.py --images_dir ./images --out_dir ./data --val_ratio 0.2

# 2) เทรนโมเดล
python train.py --csv_train data/train.csv --csv_val data/val.csv --epochs 8 --batch_size 32 --lr 3e-4 --out_dir outputs

# 3) ทดสอบทำนายภาพเดี่ยว
python predict.py --ckpt outputs/best_model.pt --image path/to/test.jpg
""",
    language="bash",
)

# --------------------------
# นำไปใช้/ต่อยอด
# --------------------------
st.header("นำไปทำอะไร/ต่อยอดได้บ้าง?")
st.markdown(
    """
- **แอปลดน้ำหนัก/รักสุขภาพ**: ให้ผู้ใช้ถ่ายรูปอาหารแล้วระบบบอกว่า *Healthy/Unhealthy* พร้อมทิปส์ปรับเมนู  
- **เมนูร้านอาหาร/คาเฟ่**: ติดป้ายสุขภาพอัตโนมัติจากรูปเมนู ช่วยทำป้ายโภชนาการเบื้องต้น  
- **ระบบแนะนำ (Recommender)**: แนะนำเมนูสุขภาพตามพฤติกรรมผู้ใช้ + เป้าหมายน้ำหนัก  
- **ต่อยอดเป็น Multi-Class**: เพิ่มคลาสอาหาร 10–50 ประเภท พร้อมคะแนนแคลอรี่โดยประมาณ  
- **อุตสาหกรรมเคาน์เตอร์แคชเชียร์ (POS/CCTV)**: ตรวจประเภทอาหารแบบเรียลไทม์เพื่อเก็บสถิติลูกค้า  
- **วิจัย/การศึกษา**: เปรียบเทียบโมเดลเบา (MobileNet/Vit-Tiny) สำหรับรันบนมือถือ  
- **MLOps**: ทำ Pipeline เต็มรูปแบบ (Data → Train → Evaluate → Deploy) + Monitor ค่า Accuracy และ Drift
"""
)

st.success("พร้อมพรีเซนต์: เปิดทีละหัวข้อ, โชว์โค้ดบล็อกสำคัญ, ปิดท้ายด้วยสรุปการใช้งานจริง 👍")
