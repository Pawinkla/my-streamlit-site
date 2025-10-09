# -*- coding: utf-8 -*-
import io
import base64
from typing import Tuple, Dict, Optional

import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# =============== ส่วนโหลดโมเดลทนทาน ===============
@st.cache_resource
def load_food_model(model_path: str = "model/best_model.pt",
                    device_str: str = "cpu") -> Tuple[nn.Module, torch.device]:
    device = torch.device(device_str)

    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, 2)  # 2 คลาส: Healthy / Unhealthy
    backbone.to(device)
    backbone.eval()

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device).eval()
        return model, device

    if isinstance(ckpt, dict):
        state = None
        for k in ["state_dict", "model_state", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            state = ckpt

        from collections import OrderedDict
        def strip_module(sd):
            new_sd = OrderedDict()
            for k, v in sd.items():
                new_sd[k[7:]] = v if k.startswith("module.") else v
            return new_sd

        try:
            backbone.load_state_dict(state)
        except Exception:
            try:
                backbone.load_state_dict(strip_module(state))
            except Exception:
                backbone.load_state_dict(strip_module(state), strict=False)

        return backbone.eval(), device

    raise RuntimeError("ไม่สามารถอ่านไฟล์โมเดลได้: รูปแบบเช็คพอยต์ไม่รองรับ")


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def predict_image(model: nn.Module, device: torch.device, img: Image.Image) -> Dict[str, float]:
    tfm = get_transform()
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    labels = ["Healthy", "Unhealthy"]
    return {labels[i]: float(probs[i]) for i in range(2)}


# =============== ส่วน OpenAI สำหรับแคลอรี่ ===============
def get_openai_key_from_secrets() -> Optional[str]:
    # ใส่คีย์ใน Streamlit Secrets เป็น OPENAI_API_KEY = "sk-xxxx"
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    return None


def image_to_data_url(img: Image.Image) -> str:
    """แปลงรูปเป็น data URL base64 เพื่อนำไปให้ GPT vision"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def estimate_calories_with_gpt(img: Image.Image, api_key: str) -> Optional[dict]:
    """
    เรียก OpenAI (GPT-4o-mini) ให้ประเมินแคลอรี่โดยดูจากรูป
    คืน dict เช่น {"calories": 620, "explanation": "..."} ถ้าสำเร็จ, ถ้าไม่สำเร็จคืน None
    """
    try:
        from openai import OpenAI
    except Exception as e:
        st.error("ไม่พบแพ็กเกจ openai โปรดเพิ่ม 'openai>=1.42.0' ใน requirements.txt")
        return None

    client = OpenAI(api_key=api_key)
    data_url = image_to_data_url(img)

    # ขอคำตอบเป็น JSON อ่านง่าย
    system_prompt = (
        "You are a helpful nutritionist. Estimate total calories of the dish in the image. "
        "Return concise JSON with keys: calories (integer kcal), explanation (short text). "
        "Assume 1 serving in the photo."
    )
    user_text = "Please estimate calories for this single-plate dish."

    try:
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=350,
        )
        content = resp.choices[0].message.content

        # พยายามดึง JSON ออกมา
        import json, re
        # หา block JSON
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            return json.loads(m.group(0))
        # ถ้าไม่มี JSON ชัดเจน แยกแบบง่าย ๆ
        return {"calories": None, "explanation": content.strip()}
    except Exception as e:
        st.warning(f"เรียก GPT ไม่สำเร็จ: {e}")
        return None


# =============== UI ===============
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

    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    # ทำนาย Healthy / Unhealthy
    probs = predict_image(model, device, img)
    pred_label = max(probs, key=probs.get)
    pred_prob = probs[pred_label]

    st.subheader("ผลลัพธ์")
    st.write(f"คำตอบ: **{pred_label}**")
    st.caption(f"ความมั่นใจ: {pred_prob*100:.2f}%")

    st.markdown("---")
    st.write("**Probability ของแต่ละคลาส**")
    st.progress(int(probs["Healthy"] * 100))
    st.write(f"Healthy: {probs['Healthy']*100:.2f}%")
    st.progress(int(probs["Unhealthy"] * 100))
    st.write(f"Unhealthy: {probs['Unhealthy']*100:.2f}%")

    # ===== แสดงส่วนคำนวณแคลอรี่ด้วย GPT (ถ้ามีคีย์) =====
    api_key = get_openai_key_from_secrets()
    if api_key:
        st.markdown("---")
        st.subheader("คำนวณแคลอรี่ (ทดลอง) ⚡")
        if st.button("ประมาณแคลอรี่ด้วย GPT"):
            with st.spinner("กำลังประเมินแคลอรี่จากรูป..."):
                cal = estimate_calories_with_gpt(img, api_key)
            if cal is None:
                st.error("ประเมินแคลอรี่ไม่สำเร็จ")
            else:
                kcal = cal.get("calories", None)
                if isinstance(kcal, (int, float)):
                    st.success(f"แคลอรี่โดยประมาณ: **{int(kcal)} kcal**")
                else:
                    st.info("ไม่พบตัวเลขแคลอรี่ชัดเจนจากคำตอบ")

                explain = cal.get("explanation")
                if explain:
                    st.caption(explain)
    else:
        st.info("ต้องใส่ `OPENAI_API_KEY` ใน Secrets ก่อน จึงจะใช้ฟีเจอร์คำนวณแคลอรี่ได้")
else:
    st.info("อัปโหลดรูปอาหารเพื่อให้โมเดลช่วยทำนาย และกดปุ่มคำนวณแคลอรี่ได้เลย 🙌")
