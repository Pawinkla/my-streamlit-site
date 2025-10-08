# -*- coding: utf-8 -*-
# Healthy vs Junk Food — Flip + Manual Mapping
import os
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")

# ---------- Config ----------
MODEL_PATH = "model/best_model.pt"
CLASS_NAMES_DEFAULT = ["Healthy", "Unhealthy"]

# ---------- Loaders ----------
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

# ---------- UI ----------
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

# 1) Toggle สลับลำดับ (เผื่อโมเดลบันทึกลำดับสลับมา)
flip = st.toggle("กลับด้านคลาส (Flip classes)", value=False,
                 help="ถ้าผลเหมือนสลับ Healthy/Unhealthy ให้เปิดสวิตช์นี้")

# 2) กำหนด mapping เองแบบชัวร์ ๆ
mapping_choice = st.radio(
    "กำหนดว่า index ไหนคือ Healthy (ควบคุมเองแบบชัวร์สุด)",
    options=["index 0 คือ Healthy", "index 1 คือ Healthy"],
    index=0,  # ค่าเริ่มต้นตามโมเดล/ความคุ้นเคย
    horizontal=True,
)

img_file = st.file_uploader("อัปโหลดรูปอาหาร (JPG/PNG)", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    # --- พยากรณ์แบบ 'ดิบ' ตามลำดับที่โมเดลให้มา ---
    prob_raw = predict(model, img, device)  # [p0, p1]
    names_raw = class_names.copy()

    # --- ขั้นที่ 1: flip ถ้าจำเป็น ---
    prob = prob_raw
    names = names_raw
    if flip:
        prob = prob[::-1]
        names = names[::-1]

    # --- ขั้นที่ 2: บังคับ mapping ตามที่ผู้ใช้เลือก ---
    # อยากให้ Healthy เป็น index ไหน ก็หมุนให้ตรง
    # หลังจบขั้นนี้ 'names[0]' จะเป็น Healthy เสมอ
    if mapping_choice == "index 1 คือ Healthy":
        if names[0] != "Healthy":
            prob = prob[::-1]
            names = names[::-1]
    else:  # "index 0 คือ Healthy"
        if names[0] != "Healthy":
            prob = prob[::-1]
            names = names[::-1]

    # --- สรุปผล ---
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

    # Debug panel (เผื่ออยากดูค่าดิบ)
    with st.expander("รายละเอียด/ค่าดิบที่โมเดลให้มา (สำหรับตรวจสอบ)"):
        st.write("class_names (ดิบจากโมเดล):", names_raw)
        st.write("prob (ดิบจากโมเดล):", {names_raw[i]: float(p) for i, p in enumerate(prob_raw)})
        st.write("class_names (หลังปรับตามตัวเลือก):", names)
        st.write("prob (หลังปรับตามตัวเลือก):", {names[i]: float(p) for i, p in enumerate(prob)})

else:
    st.info("ลาก–วาง หรือกดเลือกไฟล์ เพื่อทำนายสุขภาพของอาหารจากรูปภาพ 📸")
