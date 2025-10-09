# -*- coding: utf-8 -*-
# 📘 หน้าโชว์โค้ดโปรเจกต์ Healthy vs Junk Food (ฝังโค้ดไว้ตรง ๆ ไม่อ่านจากไฟล์)
# วางไฟล์นี้ในโฟลเดอร์ pages/ ของแอป Streamlit

import streamlit as st

st.set_page_config(page_title="โชว์โค้ด Healthy vs Junk Food", page_icon="📘", layout="wide")
st.title("📘 โชว์โค้ดโปรเจกต์: Healthy vs Junk Food")
st.caption("4 ส่วนหลักที่ใช้จริง + โค้ดสำคัญ พร้อมแนวคิดและคำสั่งรัน")

# -------------------------------------------------------------------
# 1) RULE-BASED MAPPING
# -------------------------------------------------------------------
st.header("1) กฎจัดกลุ่มอาหาร (Rule-based Mapping)")
st.markdown(
    "- เป้าหมาย: แปลง **ชื่อโฟลเดอร์/คลาสดิบ** เป็นป้ายกำกับ `Healthy / Unhealthy`\n"
    "- แนวคิด: ถ้าเจอคีย์เวิร์ดแบบทอด/หวาน/มัน → Unhealthy, ถ้าเจอคีย์เวิร์ดนึ่ง/ย่าง/ผักเยอะ → Healthy\n"
    "- ถ้าไม่เข้าเงื่อนไข ให้ **ค่าเริ่มต้น = Healthy** เพื่อเลี่ยง false positive ว่าเป็น junk"
)

code_mapping = r'''
# mapping_rules.py — ฟังก์ชันหลัก
def map_class_to_label(cls_name: str) -> str:
    c = cls_name.lower().strip()

    # (ตัวอย่าง) ลิสต์ชื่อเมนูที่ถือว่าไม่เฮลตี้
    JUNK_LIST = {"fried_chicken","pizza","burger","donut","เค้ก","ข้าวผัด","ผัดไทย","บิงซู","ชาเย็น","ชานมไข่มุก"}
    # (ตัวอย่าง) ลิสต์ชื่อเมนูที่ถือว่าเฮลตี้
    HEALTHY_LIST = {"salad","caprese_salad","beet_salad","ปลานึ่ง","ปลาเผา","ไก่ย่าง","ยำวุ้นเส้น","ต้มยำ","แกงจืด"}

    # คีย์เวิร์ดเสริม
    JUNK_KEYWORDS = ["fried","deep_fried","fries","donut","cake","sugar","butter","ทอด","ชุบแป้ง","หวาน","มันเยิ้ม","ชานม"]
    HEALTHY_KEYWORDS = ["salad","grilled","steamed","boiled","soup","ย่าง","นึ่ง","ลวก","แกงจืด","ผักเยอะ"]

    if c in JUNK_LIST:            return "Unhealthy"
    if c in HEALTHY_LIST:         return "Healthy"
    if any(k in c for k in JUNK_KEYWORDS):     return "Unhealthy"
    if any(k in c for k in HEALTHY_KEYWORDS):  return "Healthy"
    return "Healthy"  # ค่าเริ่มต้น
'''
st.code(code_mapping, language="python")

# -------------------------------------------------------------------
# 2) DATA PREP → CSV
# -------------------------------------------------------------------
st.header("2) เตรียมข้อมูล: สแกนรูป → สร้าง CSV (Train/Val)")
st.markdown(
    "- สแกนโฟลเดอร์ `images/คลาส/รูปภาพ` แล้วแม็ปเป็น `filepath,label`\n"
    "- แบ่ง **Train/Validation** แบบ stratified keeping class ratio\n"
    "- ได้ไฟล์ `data/train.csv` และ `data/val.csv`"
)

code_prep = r'''
# prepare_dataset.py — แปลงโฟลเดอร์รูปเป็น CSV
import os, csv, random
from typing import List, Tuple

def scan_images(images_dir: str) -> List[Tuple[str, str]]:
    items = []
    for cls in sorted(os.listdir(images_dir)):
        cdir = os.path.join(images_dir, cls)
        if not os.path.isdir(cdir):
            continue
        label = map_class_to_label(cls)  # เรียกใช้กฎจากข้อ (1)
        for root, _, files in os.walk(cdir):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    items.append((os.path.join(root, f), label))
    return items

def stratified_split(items, val_ratio=0.2, seed=42):
    random.seed(seed)
    by = {}
    for p,l in items:
        by.setdefault(l, []).append((p,l))
    tr, va = [], []
    for l, arr in by.items():
        random.shuffle(arr)
        n_val = max(1, int(len(arr) * val_ratio))
        va += arr[:n_val]
        tr += arr[n_val:]
    random.shuffle(tr); random.shuffle(va)
    return tr, va

def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label"])
        w.writerows(rows)

# วิธีใช้:
# items = scan_images("./images")
# tr, va = stratified_split(items, 0.2)
# write_csv(tr, "data/train.csv"); write_csv(va, "data/val.csv")
'''
st.code(code_prep, language="python")

# -------------------------------------------------------------------
# 3) TRAINING — ResNet-18
# -------------------------------------------------------------------
st.header("3) เทรนโมเดล ResNet-18 (Fine-tune)")
st.markdown(
    "- โหลด **ResNet-18 (ImageNet weights)** → ปรับชั้นสุดท้ายเป็น 2 คลาส\n"
    "- ใช้ **CrossEntropyLoss + Adam**\n"
    "- เซฟ `outputs/best_model.pt` เมื่อค่า **val acc** ดีขึ้น"
)

code_train = r'''
# train.py — ส่วนสำคัญของการเทรน
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd, os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class CsvImageDataset(Dataset):
    def __init__(self, csv_path, tfm):
        df = pd.read_csv(csv_path)
        self.paths = df["filepath"].tolist()
        labels = df["label"].tolist()
        names = sorted(list(set(labels)))
        self.name_to_idx = {n:i for i,n in enumerate(names)}
        self.idx_to_name = names
        self.labels = [self.name_to_idx[l] for l in labels]
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]

def build_loaders(csv_train, csv_val, bs=32):
    tfm_tr = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tfm_va = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds_tr = CsvImageDataset(csv_train, tfm_tr)
    ds_va = CsvImageDataset(csv_val, tfm_va)
    return (
        DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
        ds_tr.idx_to_name
    )

def train(csv_train, csv_val, epochs=8, lr=3e-4, out_dir="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_tr, loader_va, class_names = build_loaders(csv_train, csv_val, 32)

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(class_names)),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc, best_path = 0.0, os.path.join(out_dir, "best_model.pt")
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        # train
        model.train(); tot, corr, loss_sum = 0, 0, 0.0
        for x, y in loader_tr:
            x, y = x.to(device), torch.tensor(y, dtype=torch.long, device=device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            loss_sum += float(loss) * x.size(0)
            corr += (out.argmax(1)==y).sum().item()
            tot  += x.size(0)
        tr_loss, tr_acc = loss_sum/tot, corr/tot

        # validate
        model.eval(); tot, corr, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader_va:
                x, y = x.to(device), torch.tensor(y, dtype=torch.long, device=device)
                out = model(x); loss = criterion(out, y)
                loss_sum += float(loss) * x.size(0)
                corr += (out.argmax(1)==y).sum().item()
                tot  += x.size(0)
        va_loss, va_acc = loss_sum/tot, corr/tot

        print(f"Epoch {epoch}/{epochs} | train {tr_acc:.3f} | val {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            }, best_path)
            print(f"Saved best: {best_path} (acc={best_acc:.3f})")

# วิธีใช้:
# train('data/train.csv','data/val.csv', epochs=8, lr=3e-4, out_dir='outputs')
'''
st.code(code_train, language="python")

# -------------------------------------------------------------------
# 4) INFERENCE — ใช้งานโมเดล
# -------------------------------------------------------------------
st.header("4) นำโมเดลมาใช้งาน (Predict/Inference)")
st.markdown(
    "- โหลด `best_model.pt` → เตรียมรูป 224×224 → softmax แล้วเลือกคลาสที่ความน่าจะเป็นสูงสุด\n"
    "- ใช้ได้ทั้ง CLI, batch script หรือฝังในหน้า Streamlit อื่น"
)

code_predict = r'''
# predict.py — โหลดเช็คพอยต์แล้วทำนายภาพเดี่ยว
import torch
from PIL import Image
from torchvision import transforms, models

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, len(ckpt["class_names"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["class_names"], ckpt["mean"], ckpt["std"]

def predict_image(img_path, ckpt_path="outputs/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, mean, std = load_model(ckpt_path, device)
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(prob.argmax())
    return names[idx], float(prob[idx])

# ตัวอย่าง:
# label, p = predict_image("path/to/test.jpg", "outputs/best_model.pt")
# print(label, p)
'''
st.code(code_predict, language="python")

# -------------------------------------------------------------------
# คำสั่งที่ใช้จริง
# -------------------------------------------------------------------
st.subheader("คำสั่งที่ใช้จริง (รันบนเทอร์มินัล)")
st.code("""\
# 1) เตรียม CSV
python prepare_dataset.py --images_dir ./images --out_dir ./data --val_ratio 0.2

# 2) เทรนโมเดล
python train.py --csv_train data/train.csv --csv_val data/val.csv --epochs 8 --batch_size 32 --lr 3e-4 --out_dir outputs

# 3) ทดสอบทำนาย
python predict.py --ckpt outputs/best_model.pt --image path/to/test.jpg
""", language="bash")

# -------------------------------------------------------------------
# ต่อยอดใช้งาน
# -------------------------------------------------------------------
st.header("โปรเจกต์นี้นำไปทำอะไร / ต่อยอดอะไรได้บ้าง?")
st.markdown("""
- **แอปช่วยกินเฮลตี้**: ผู้ใช้ถ่ายรูปอาหาร ระบบบอกว่า Healthy/Unhealthy พร้อมคำแนะนำปรับเมนู
- **POS/CCTV ร้านอาหาร**: ติดกล้องหน้าครัว/แคชเชียร์ เพื่อนับสถิติเครื่องดื่มหวาน/ของทอดแบบเรียลไทม์
- **เมนูดิจิทัล**: ติดป้ายสุขภาพอัตโนมัติ ใส่คะแนนความเฮลตี้ + คำเตือนน้ำตาล/ไขมัน
- **Recommender ส่วนบุคคล**: แนะนำเมนูที่สอดคล้องกับเป้าหมายน้ำหนัก/แคลอรี่ของผู้ใช้
- **ขยายเป็น Multi-Class**: จำแนกชนิดอาหาร 10–50 คลาส พร้อมประเมินแคลอรี่โดยประมาณ
- **รันบนมือถือ**: แปลงเป็นโมเดลเบา (MobileNet/Vit-Tiny, TorchScript/ONNX) ใช้งานออฟไลน์
- **MLOps เต็มระบบ**: Pipeline ดึงข้อมูล → เทรน → ประเมิน → ดีพลอย → มอนิเตอร์ model drift
""")

st.success("พร้อมพรีเซนต์แล้ว! เปิดทีละหัวข้อ โชว์โค้ดบล็อกสำคัญ แล้วปิดด้วยการต่อยอด 👍")
