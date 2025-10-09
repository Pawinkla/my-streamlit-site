# -*- coding: utf-8 -*-
import io
from typing import Tuple, Dict

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ---------------------------
# 1) ตัวโหลดโมเดล (ทนทานต่อรูปแบบเช็คพอยต์หลายแบบ)
# ---------------------------
@st.cache_resource
def load_food_model(model_path: str = "model/best_model.pt",
                    device_str: str = "cpu") -> Tuple[nn.Module, torch.device]:
    """
    โหลดโมเดล ResNet18 ที่หัวรองรับ 2 classes (Healthy/Unhealthy)
    รองรับไฟล์ .pt ที่เป็นทั้ง state_dict และโมเดลเต็ม ๆ
    รวมถึงกรณีคีย์พารามิเตอร์มี prefix 'module.'.
    """
    device = torch.device(device_str)

    # สถาปัตย์อ้างอิง: ResNet18 + fc 2 คลาส
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)
    backbone.to(device)
    backbone.eval()

    ckpt = torch.load(model_path, map_location=device)

    # กรณีเป็นโมเดลเต็ม ๆ
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device).eval()
        return model, device

    # กรณีเป็น dict: พยายามหา state_dict ภายใต้คีย์ยอดนิยม
    if isinstance(ckpt, dict):
        state = None
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            state = ckpt  # เป็น dict ของพารามิเตอร์อยู่แล้ว

        # ตัด prefix 'module.' หากมี
        from collections import OrderedDict
        def strip_module(sd):
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_sd[k[7:]] = v if k.startswith("module.") else v
            return new_sd

        try:
            backbone.load_state_dict(state)  # strict=True ก่อน
        except Exception:
            try:
                backbone.load_state_dict(strip_module(state))
            except Exception:
                # ทางเลือกสุดท้ายให้ strict=False
                backbone.load_state_dict(strip_module(state), strict=False)

        return backbone.eval(), device

    # ถ้าเข้ามาตรงนี้ แปลว่ารูปแบบไฟล์แปลกมาก
    raise RuntimeError("ไม่สามารถอ่านไฟล์โมเดลได้: รูปแบบเช็คพอยต์ไม่รองรับ")


# ---------------------------
# 2) Preprocess สำหรับภาพ
# ---------------------------
@st.cache_resource
def get_transform():
    # เหมือนที่ใช้ตอนเทรนทั่วไปของ ResNet18 (ImageNet-style)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def predict_image(model: nn.Module, device: torch.device, img: Image.Image) -> Dict[str, float]:
    """
    รับ PIL Image -> คืนความน่าจะเป็นของสองคลาสเป็น dict
    """
    tfm = get_transform()
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    # ลำดับคลาสตามหัวโมเดล: index 0 = Healthy, index 1 = Unhealthy
    labels = ["Healthy", "Unhealthy"]
    return {labels[i]: float(probs[i]) for i in range(2)}


# ---------------------------
# 3) UI
# ---------------------------
st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")

st.title("Healthy vs Junk Food 🥗")

# โหลดโมเดล
try:
    model, device = load_food_model("model/best_model.pt", device_str="cpu")
    st.success("โหลดโมเดลสำเร็จจาก `model/best_model.pt` (อุปกรณ์: CPU)")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# อัปโหลดภาพ
st.subheader("อัปโหลดรูปอาหาร (JPG/PNG)")
uploaded = st.file_uploader(
    "Drag & drop หรือกด *Browse files* เพื่อเลือกรูป",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception:
        st.error("ไม่สามารถเปิดรูปได้ ลองไฟล์อื่นอีกครั้ง")
        st.stop()

    # แสดงภาพ
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    # พยากรณ์
    probs = predict_image(model, device, img)
    pred_label = max(probs, key=probs.get)
    pred_prob = probs[pred_label]

    st.subheader("ผลลัพธ์")
    st.write(f"คำตอบ: **{pred_label}**")
    st.caption(f"ความมั่นใจ: {pred_prob*100:.2f}%")

    # แสดงความน่าจะเป็นแบบแยกคลาส
    st.markdown("---")
    st.write("**Probability ของแต่ละคลาส**")
    st.progress(int(probs["Healthy"] * 100))
    st.write(f"Healthy: {probs['Healthy']*100:.2f}%")

    st.progress(int(probs["Unhealthy"] * 100))
    st.write(f"Unhealthy: {probs['Unhealthy']*100:.2f}%")
else:
    st.info("อัปโหลดรูปอาหารเพื่อให้โมเดลช่วยทำนายได้เลย 🙌")
