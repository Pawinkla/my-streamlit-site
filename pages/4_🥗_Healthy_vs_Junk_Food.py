# -*- coding: utf-8 -*-
import io
import json
import base64
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ========== CONFIG ==========
# เก็บคีย์ไว้ใน Secrets ของ Streamlit Cloud: OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GPT_MODEL = "gpt-4o-mini"   # โมเดล Vision ของ OpenAI
# ============================

st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗", layout="centered")
st.title("Healthy vs Junk Food 🥗")

# ------------ โหลดโมเดลของคุณ (ResNet18 2-class) ------------
@st.cache_resource
def load_food_model(model_path="model/best_model.pt", device="cpu"):
    device = torch.device(device)
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: Healthy / Unhealthy
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, device

model, device = load_food_model()

# ------------ Transform -------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])

# ------------ ฟังก์ชัน inference -------------
@torch.inference_mode()
def predict(img: Image.Image):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    idx = int(probs.argmax())
    classes = ["Healthy", "Unhealthy"]
    return classes[idx], float(probs[0]), float(probs[1])

# ========== ส่วน GPT: ประมาณแคลอรี่จากรูป ==========
def _to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def estimate_calories_with_gpt(image: Image.Image, detail_hint: str = ""):
    """
    เรียก OpenAI Vision ให้ตอบเป็น JSON:
      { "calories_kcal": number, "confidence": 0-1, "items": [ { "name": str, "kcal": number } ] }
    """
    import openai  # ไลบรารี openai (เวอร์ชัน ChatCompletion)
    openai.api_key = OPENAI_API_KEY

    img_b64 = _to_base64(image)
    system_prompt = (
        "You are a nutrition assistant. "
        "Given a meal photo, estimate the total calories (kcal). "
        "List key items with rough kcal breakdown. "
        "Respond ONLY in JSON with keys: calories_kcal, confidence, items[]."
    )
    user_prompt = (
        "Estimate calories of this meal. If uncertain, give your best reasonable guess."
        + (f" Extra context: {detail_hint}" if detail_hint else "")
    )

    data_url = f"data:image/jpeg;base64,{img_b64}"

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    text = completion.choices[0].message["content"]
    # พยายาม parse JSON จากข้อความตอบ
    try:
        if "```" in text:  # เผื่อโมเดลส่งใน code fence
            text = text.split("```", 2)[1]
            if text.lower().startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return {
            "ok": True,
            "calories_kcal": float(data.get("calories_kcal", 0)),
            "confidence": float(data.get("confidence", 0)),
            "items": data.get("items", []),
            "raw": text,
        }
    except Exception as e:
        return {"ok": False, "error": f"Cannot parse JSON: {e}", "raw": text}

# ========== UI ==========

uploaded = st.file_uploader("อัปโหลดรูปอาหาร (JPG/PNG)", type=["jpg", "jpeg", "png"])
hint = st.text_input("(ไม่จำเป็น) ใส่คำใบ้ให้ GPT เช่น 'อกไก่ย่าง อะโวคาโด ผักสลัด น้ำสลัดงาญี่ปุ่น'", "")

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="ภาพที่อัปโหลด", use_column_width=True)

    label, p_healthy, p_unhealthy = predict(img)
    st.subheader("ผลลัพธ์")
    st.markdown(f"**คำตอบ:** {label}")
    st.caption(f"ความมั่นใจ (Healthy): {p_healthy*100:.2f}% — (Unhealthy): {p_unhealthy*100:.2f}%")

    # ===== แสดงแคลอรี่ด้วย GPT ใต้ผลลัพธ์ =====
    with st.spinner("ประมาณแคลอรี่ด้วย GPT…"):
        g = estimate_calories_with_gpt(img, hint.strip())
    if g.get("ok"):
        st.markdown("### 🔥 ประมาณแคลอรี่ (GPT)")
        st.markdown(
            f"**ประมาณ:** ~ **{g['calories_kcal']:.0f} kcal**  \n"
            f"**ความมั่นใจ (GPT):** {g['confidence']*100:.1f}%"
        )
        if g.get("items"):
            st.markdown("**รายการหลัก (ประมาณ):**")
            for it in g["items"]:
                name = it.get("name", "item")
                kcal = it.get("kcal", None)
                st.markdown(f"- {name}" + (f": ~{kcal:.0f} kcal" if kcal is not None else ""))

        with st.expander("ผลดิบจาก GPT (JSON)"):
            st.code(json.dumps({
                "calories_kcal": g["calories_kcal"],
                "confidence": g["confidence"],
                "items": g["items"],
            }, ensure_ascii=False, indent=2), language="json")
    else:
        st.error("ประเมินแคลอรี่ล้มเหลว")
        st.caption(g.get("error", ""))
        with st.expander("ข้อความตอบกลับจาก GPT"):
            st.code(g.get("raw", ""), language="json")
else:
    st.info("ลาก-วาง หรือเลือกไฟล์ เพื่อวิเคราะห์สุขภาพของอาหารและประมาณแคลอรี่จากภาพ")
