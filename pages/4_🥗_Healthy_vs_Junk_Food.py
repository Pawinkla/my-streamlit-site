# -*- coding: utf-8 -*-
# Healthy vs Junk Food – หน้าใหม่ (โหลดโมเดลอัตโนมัติจาก model/best_model.pt)
import os, io
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# ---------- Page config ----------
st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")

# ---------- Constants ----------
MODEL_PATH = "model/best_model.pt"   # <- เปลี่ยนที่นี่ถ้าคุณย้ายไฟล์
CLASS_NAMES_DEFAULT = ["Healthy", "Unhealthy"]

# ---------- Utils ----------
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
    # รองรับทั้ง state dict ตรง ๆ หรือบันทึกใน key "model"
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
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    return prob

# ---------- UI ----------
st.markdown("## 🥗 Healthy vs Junk Food  \nอัปโหลดรูป → ได้ผลทันที (ไม่ต้องตั้งเกณฑ์)")
st.caption("โมเดลจะถูกโหลดอัตโนมัติจาก `model/best_model.pt`")

# โหลดโมเดล
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, class_names = load_model(MODEL_PATH, device)
    st.success(f"โหลดโมเดลสำเร็จ ✅  (อุปกรณ์ที่ใช้: {device.upper()})")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# อัปโหลดรูป
img_file = st.file_uploader("อัปโหลดรูปอาหาร (JPG/PNG)", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file).convert("RGB")

    # แสดงรูปใหญ่สวย ๆ
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    # พยากรณ์
    prob = predict(model, img, device)
    top_idx = int(prob.argmax())
    label = class_names[top_idx] if 0 <= top_idx < len(class_names) else f"class_{top_idx}"
    conf = float(prob[top_idx])

    st.markdown("---")
    st.markdown(f"### ผลลัพธ์: **{label}**")
    st.caption(f"ความมั่นใจ: **{conf:.2f}**")

    # แสดง prob สองคลาสแบบแท่ง
    c1, c2 = st.columns(2)
    with c1:
        st.metric(class_names[0] if len(class_names) > 0 else "Healthy", f"{prob[0]:.2f}")
        st.progress(min(max(float(prob[0]), 0.0), 1.0))
    with c2:
        st.metric(class_names[1] if len(class_names) > 1 else "Unhealthy", f"{prob[1]:.2f}")
        st.progress(min(max(float(prob[1]), 0.0), 1.0))

    st.markdown("—")
    st.json({class_names[i] if i < len(class_names) else f"class_{i}": float(p) for i, p in enumerate(prob)})

else:
    st.info("ลาก–วาง หรือกดเลือกไฟล์ เพื่อทำนายสุขภาพของอาหารจากรูปภาพ 📸")
