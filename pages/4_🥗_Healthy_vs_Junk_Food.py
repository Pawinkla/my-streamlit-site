# -*- coding: utf-8 -*-
# Healthy vs Junk Food — Simplified (no toggles, no class pickers)
import os
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")

MODEL_PATH = "model/best_model.pt"
CLASS_NAMES_DEFAULT = ["Healthy", "Unhealthy"]

@st.cache_resource
def load_model(path: str, device: str = "cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ '{path}'")

    ckpt = torch.load(path, map_location=device)

    # โมเดลโครง ResNet18 (หัว 2 คลาส)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    # รองรับทั้ง ckpt เป็น state_dict ตรง ๆ หรือ dict ที่มีคีย์ "model"
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # ถ้ามี class_names ใน ckpt ใช้เลย ไม่งั้น fallback
    class_names = (
        ckpt.get("class_names", CLASS_NAMES_DEFAULT)
        if isinstance(ckpt, dict) else CLASS_NAMES_DEFAULT
    )
    # ปรับให้เป็น list[str]
    class_names = list(map(str, class_names))
    return model, class_names

# Transform ให้ตรงกับตอนเทรน (ImageNet mean/std)
TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(model, img: Image.Image, device: str = "cpu") -> np.ndarray:
    x = TFM(img.convert("RGB")).unsqueeze(0)
    if device == "cuda":
        x = x.to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].detach().cpu().numpy()
    return prob  # shape (2,)

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

# อัปโหลดรูปเพื่อทำนาย
img_file = st.file_uploader("อัปโหลดรูปอาหาร (JPG/PNG)", type=["jpg", "jpeg", "png"])
if not img_file:
    st.info("ลาก–วาง หรือกดเลือกไฟล์ เพื่อทำนายสุขภาพของอาหารจากรูปภาพ 📸")
    st.stop()

img = Image.open(img_file).convert("RGB")
st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

# --- ค่าดิบจากโมเดล ---
prob_raw = predict(model, img, device)     # [p0, p1]
names_raw = class_names[:]                 # ตามเช็คพอยต์เดิม

# --- จัดระเบียบชื่อคลาส/ความน่าจะเป็นแบบอัตโนมัติ ---
names = names_raw[:]
prob = prob_raw.copy()

# ถ้าเช็คพอยต์มีชื่อ 'Healthy' และ 'Unhealthy' ให้จัดลำดับให้ Healthy อยู่ซ้ายเสมอ
if set(["Healthy", "Unhealthy"]).issubset(set(n.lower().capitalize() for n in names_raw)):
    # ทำให้ชื่อมีฟอร์แมตตรงกันก่อน
    names_norm = [n.lower().capitalize() for n in names_raw]
    healthy_idx = names_norm.index("Healthy")
    unhealthy_idx = names_norm.index("Unhealthy")
    order = [healthy_idx, unhealthy_idx]
    names = [names_raw[i] for i in order]
    prob = prob_raw[order]
# ไม่งั้น ให้คงลำดับจากโมเดลตามเดิม

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
    st.write("prob (ดิบจากโมเดล):", {names_raw[i]: float(prob_raw[i]) for i in range(len(prob_raw))})
    st.write("class_names (หลังจัดลำดับ):", names)
    st.write("prob (หลังจัดลำดับ):", {names[i]: float(prob[i]) for i in range(len(prob))})
