# -*- coding: utf-8 -*-
# Healthy vs Junk Food — Flip + Choose Healthy label robustly
import os
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")

MODEL_PATH = "model/best_model.pt"
CLASS_NAMES_DEFAULT = ["Healthy", "Unhealthy"]

@st.cache_resource
def load_model(path: str, device: str = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ '{path}'")

    ckpt = torch.load(path, map_location=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval().to(device)

    class_names = ckpt.get("class_names", CLASS_NAMES_DEFAULT) if isinstance(ckpt, dict) else CLASS_NAMES_DEFAULT
    return model, class_names

TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(model, img: Image.Image, device: str = "cpu"):
    x = TFM(img.convert("RGB")).unsqueeze(0)
    if device == "cuda":
        x = x.to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return prob  # [p0, p1]

# ---------------- UI ----------------
st.markdown("## 🥗 Healthy vs Junk Food")
st.caption("โหลดโมเดลอัตโนมัติจาก `model/best_model.pt` แล้วอัปโหลดภาพเพื่อทำนายได้ทันที")

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, class_names = load_model(MODEL_PATH, device)
    st.success(f"โหลดโมเดลสำเร็จ ✅ (อุปกรณ์: {device.upper()})")
    st.caption(f"ลำดับคลาสจากโมเดล → index 0: **{class_names[0]}**, index 1: **{class_names[1]}**")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# 1) Toggle กลับด้านคลาส (สลับ index ทั้งชุด)
flip = st.toggle("กลับด้านคลาส (Flip classes)", value=False,
                 help="ถ้าผลเหมือนสลับ Healthy/Unhealthy ให้เปิดสวิตช์นี้")

# 2) เลือกด้วย “ชื่อคลาสจริง” ว่าอันไหนคือ Healthy (กันพลาด 100%)
healthy_label_pick = st.radio(
    "กำหนดว่า **คำไหน** คือ Healthy (ควบคุมเองแบบชัวร์สุด)",
    options=class_names,   # ให้เลือกจากชื่อในโมเดลเลย
    index=0,
    horizontal=True,
)

img_file = st.file_uploader("อัปโหลดรูปอาหาร (JPG/PNG)", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    # --- ค่าดิบจากโมเดล ---
    prob_raw = predict(model, img, device)  # [p0, p1]
    names_raw = class_names[:]              # ['Healthy','Unhealthy'] หรืออย่างอื่น

    # ทำให้ deterministic: เริ่มจากค่าดิบทุกครั้ง
    names = names_raw[:]
    prob = prob_raw.copy()

    # ขั้นที่ 1: flip ถ้าผู้ใช้สั่ง
    if flip:
        prob = prob[::-1]
        names = names[::-1]

    # ขั้นที่ 2: บังคับให้ "Healthy label" ไปอยู่ที่ index 0 เสมอ
    # ถ้าสิ่งที่ผู้ใช้เลือก (healthy_label_pick) อยู่ที่ index 1 ตอนนี้ → พลิก
    if healthy_label_pick == names[1]:
        prob = prob[::-1]
        names = names[::-1]
    # ถ้า healthy_label_pick == names[0] ก็ไม่ต้องทำอะไร

    # -------- แสดงผล --------
    top = int(prob.argmax())
    label = names[top]
    conf = float(prob[top])

    st.markdown("---")
    st.markdown(f"### ผลลัพธ์: **{label}**")
    st.caption(f"ความมั่นใจ: **{conf:.2f}**")

    c1, c2 = st.columns(2)
    with c1:
        st.metric(names[0], f"{prob[0]:.2f}")
        st.progress(min(max(float(prob[0]), 0.0), 1.0))
    with c2:
        st.metric(names[1], f"{prob[1]:.2f}")
        st.progress(min(max(float(prob[1]), 0.0), 1.0))

    with st.expander("รายละเอียด/ค่าดิบเพื่อการตรวจสอบ"):
        st.write("class_names (ดิบจากโมเดล):", names_raw)
        st.write("prob (ดิบจากโมเดล):", {names_raw[i]: float(p) for i, p in enumerate(prob_raw)})
        st.write("Healthy (ที่เลือก):", healthy_label_pick)
        st.write("class_names (หลังปรับ):", names)
        st.write("prob (หลังปรับ):", {names[i]: float(p) for i, p in enumerate(prob)})

else:
    st.info("ลาก–วาง หรือกดเลือกไฟล์ เพื่อทำนายสุขภาพของอาหารจากรูปภาพ 📸")
