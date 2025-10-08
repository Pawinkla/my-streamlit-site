
# -*- coding: utf-8 -*-
import io, os
import streamlit as st, torch
from PIL import Image
from torchvision import transforms, models
from torch import nn
st.set_page_config(page_title="Healthy vs Junk Food", page_icon="🥗")

@st.cache_resource
def load_model_from_bytes(ckpt_bytes: bytes, device: str):
    buf = io.BytesIO(ckpt_bytes); ckpt = torch.load(buf, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
    model.load_state_dict(ckpt["model"]); model.eval(); model.to(device)
    names = ckpt.get("class_names", ["Healthy","Unhealthy"]); return model, names

@st.cache_resource
def load_model_from_path(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
    model.load_state_dict(ckpt["model"]); model.eval(); model.to(device)
    names = ckpt.get("class_names", ["Healthy","Unhealthy"]); return model, names

from torchvision import transforms

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


st.title("🥗 Healthy vs Junk Food 🍟")
device = "cuda" if torch.cuda.is_available() else "cpu"

opt = st.radio("เลือกวิธีโหลดโมเดล", ["อัปโหลดไฟล์ .pt", "พาธไฟล์ในเครื่องเซิร์ฟเวอร์"], horizontal=True)
model=None; names=["Healthy","Unhealthy"]

if opt=="อัปโหลดไฟล์ .pt":
    up=st.file_uploader("อัปโหลดโมเดล (.pt)", type=["pt"])
    if up: model,names=load_model_from_bytes(up.read(),device); st.success("โหลดโมเดลแล้ว ✅")
else:
    path=st.text_input("พาธไฟล์โมเดล (.pt)", value="outputs/best_model.pt")
    if path and os.path.exists(path): model,names=load_model_from_path(path,device); st.success(f"โหลดโมเดลจาก `{path}` แล้ว ✅")
    else: st.info("ระบุพาธในเครื่อง (ตอนรัน local) หรือเลือก 'อัปโหลดไฟล์ .pt'")

st.divider()
img=st.file_uploader("อัปโหลดรูปอาหาร", type=["jpg","jpeg","png"])
thr=st.slider("เกณฑ์ความมั่นใจขั้นต่ำ (Unsure ถ้าต่ำกว่า)", 0.5, 0.95, 0.60, 0.01)
if img:
    if model is None: st.error("กรุณาโหลดโมเดลก่อน")
    else:
        im=Image.open(img).convert("RGB"); st.image(im, caption="ภาพที่อัปโหลด", use_container_width=True)
        x=tf(im).unsqueeze(0); 
        if device=="cuda": x=x.to(device)
        with torch.no_grad(): p=torch.softmax(model(x),dim=1)[0].detach().cpu().numpy()
        conf=float(p.max()); label=names[int(p.argmax())]
        st.subheader(f"ผลลัพธ์: **{'ไม่แน่ใจ' if conf<thr else label}**  (ความมั่นใจ {conf:.2f})")
        st.write({"Healthy": float(p[0]), "Unhealthy": float(p[1])}); st.progress(conf)
