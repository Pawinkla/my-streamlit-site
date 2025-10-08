# pages/4_🥗_Healthy_vs_Junk_Food.py
# -*- coding: utf-8 -*-

import os
import io
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st


# -----------------------------
# ตั้งค่าเริ่มต้น
# -----------------------------
st.set_page_config(
    page_title="Healthy vs Junk Food",
    page_icon="🥗",
    layout="centered",
)

MODEL_PATH = "model/best_model.pt"
CLASS_NAMES_DEFAULT = ["Healthy", "Unhealthy"]   # ให้ชื่อคลาสเรียงตามที่คุณเทรน


# -----------------------------
# ฟังก์ชันโหลดโมเดล (โหมดถึก)
# รองรับหลายรูปแบบการ save:
#   - torch.save(model)
#   - torch.save(model.state_dict())
#   - torch.save({'model': state_dict, 'class_names': ...})
#   - เคส DataParallel ที่คีย์ขึ้นต้น "module."
# -----------------------------
@st.cache_resource
def load_model(path: str, device: str = "cpu") -> Tuple[nn.Module, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {path}")

    ckpt = torch.load(path, map_location=device)

    # 1) ถ้า save มาเป็น "ตัวโมเดลทั้งตัว"
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        class_names = getattr(model, "class_names", CLASS_NAMES_DEFAULT)
        class_names = list(map(str, class_names))
        return model, class_names

    # 2) ถ้าเป็น dict, หา state_dict ตามคีย์ยอดนิยม
    state_dict = None
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            # น่าจะเป็น state_dict ตรง ๆ
            state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(
            "ไฟล์โมเดลไม่อยู่ในรูปแบบที่รู้จัก (nn.Module หรือ state_dict หรือ {'model': ...})"
        )

    # 3) ถ้าคีย์เป็น DataParallel (ขึ้นต้น 'module.')
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        new_sd = {}
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        state_dict = new_sd

    # 4) ประกอบสถาปัตย์ให้ตรงกับที่เทรน: ResNet18 + head 2 คลาส
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )

    # ตั้ง strict=False เพื่อยืดหยุ่น (ป้องกันบัฟเฟอร์คีย์ไม่ตรงบางตัว)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        st.warning(
            f"weights บางส่วนไม่พบในโมเดล: {missing[:8]}{' ...' if len(missing) > 8 else ''}"
        )
    if unexpected:
        st.warning(
            f"พบคีย์ส่วนเกินในไฟล์ weights: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )

    model.to(device).eval()

    # 5) class_names หากบันทึกไว้ใน checkpoint
    class_names = CLASS_NAMES_DEFAULT
    if isinstance(ckpt, dict) and "class_names" in ckpt:
        try:
            class_names = list(map(str, ckpt["class_names"]))
        except Exception:
            pass

    return model, class_names


# -----------------------------
# Transform สำหรับภาพ
# -----------------------------
def build_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# -----------------------------
# ฟังก์ชันพยากรณ์
# -----------------------------
def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: str,
) -> torch.Tensor:
    tfm = build_transform(224)
    with torch.no_grad():
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        probs = logits.softmax(dim=1).squeeze(0).cpu()
    return probs


# -----------------------------
# UI
# -----------------------------
st.title("Healthy vs Junk Food  🥗")

# โหลดโมเดล
device = "cpu"  # Streamlit Cloud มักจะเป็น CPU
try:
    model, class_names = load_model(MODEL_PATH, device=device)
    st.success(f"โหลดโมเดลสำเร็จจาก `{MODEL_PATH}` (อุปกรณ์: {device.upper()})")
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# ตัวเลือกอัปโหลดภาพ
st.markdown("### อัปโหลดรูปอาหาร (JPG/PNG)")
file = st.file_uploader(
    "Drag & drop หรือกดปุ่มเพื่อเลือกไฟล์",
    type=["jpg", "jpeg", "png"],
)

if file is not None:
    # แสดงภาพ
    image = Image.open(io.BytesIO(file.read()))
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    # พยากรณ์
    probs = predict_image(model, image, device)
    pred_idx = int(probs.argmax().item())
    pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
    confidence = float(probs[pred_idx].item())

    st.markdown("### ผลลัพธ์")
    st.subheader(f"คำตอบ: **{pred_name}**")
    st.caption(f"ความมั่นใจ: {confidence:.2%}")

    # แสดง bars ของทุกคลาส
    st.markdown("---")
    st.markdown("**Probability ของแต่ละคลาส**")
    for i, p in enumerate(probs.tolist()):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        st.write(f"{name}: {p:.2%}")
        st.progress(min(max(p, 0.0), 1.0))
else:
    st.info("อัปโหลดรูปภาพเพื่อให้โมเดลพยากรณ์")
