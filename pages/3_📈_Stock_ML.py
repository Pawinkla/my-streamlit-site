# app.py — Stock Buy / Not Buy (Thai UI, Streamlit)
# - คำนวณ revenue_growth_ttm จริงจากงบ TTM ปัจจุบันเทียบ TTM ก่อนหน้า
# - แพตช์ให้โมเดลเก่าใช้ได้กับ sklearn ใหม่ (เติม monotonic_cst)
# - normalize proba ให้อยู่ในช่วง 0..1 เสมอ
# - แสดงผลภาษาไทย

import json
import warnings
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from joblib import load
import sklearn

warnings.filterwarnings("ignore")
st.set_page_config(page_title="สแกนหุ้น: น่าซื้อ/ไม่น่าซื้อ (Fundamental ML)", layout="wide")

# =============================
# โหลดโมเดล + แพตช์ความเข้ากันได้
# =============================
def _patch_monotonic_cst(model):
    """เติม attribute 'monotonic_cst' ให้ต้นไม้ภายใน RF/ET เพื่อให้โมเดล sklearn เก่า
    ใช้กับ sklearn >=1.4 ได้"""
    try:
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(est, "estimators_"):
            for tree in est.estimators_:
                if not hasattr(tree, "monotonic_cst"):
                    setattr(tree, "monotonic_cst", None)
    except Exception:
        pass

@st.cache_resource(show_spinner=False)
def load_model_and_meta(model_path="buy_model_pipeline.pkl",
                        meta_path="model_meta.json"):
    pipe = None
    try:
        pipe = load(model_path)
        _patch_monotonic_cst(pipe)
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่ได้: {model_path}\n{e}")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        st.warning("⚠️ อ่าน model_meta.json ไม่ได้ — ใช้ค่าเริ่มต้นแทน.")
        meta = {
            "feature_cols": [
                "revenue_growth_ttm","gross_margin_ttm","operating_margin_ttm","net_margin_ttm",
                "roe_ttm","roa_ttm","debt_to_equity","ocf_to_cl","interest_coverage","asset_turnover_ttm"
            ],
            "exchange_suffix": ".BK",
            "forward_months": 6
        }
    return pipe, meta

pipe, meta = load_model_and_meta()
FEATURE_COLS: List[str] = meta.get("feature_cols", [])
SUFFIX: str = meta.get("exchange_suffix", ".BK")
FORWARD_MONTHS: int = meta.get("forward_months", 6)

# คำอธิบายฟีเจอร์ (ไทยย่อ)
FEATURE_DESC = {
    "revenue_growth_ttm": "การเติบโตของรายได้ (TTM, เทียบ TTM ก่อนหน้า)",
    "gross_margin_ttm": "อัตรากำไรขั้นต้น (TTM)",
    "operating_margin_ttm": "อัตรากำไรจากการดำเนินงาน (TTM)",
    "net_margin_ttm": "อัตรากำไรสุทธิ (TTM)",
    "roe_ttm": "ROE (กำไรสุทธิ / ส่วนของผู้ถือหุ้นเฉลี่ย)",
    "roa_ttm": "ROA (กำไรสุทธิ / สินทรัพย์รวมเฉลี่ย)",
    "debt_to_equity": "อัตราส่วนหนี้สินต่อส่วนผู้ถือหุ้น",
    "ocf_to_cl": "กระแสเงินสดดำเนินงาน (TTM) / หนี้สินหมุนเวียน",
    "interest_coverage": "ความสามารถชำระดอกเบี้ย (EBIT / |ดอกเบี้ย|)",
    "asset_turnover_ttm": "ประสิทธิภาพใช้สินทรัพย์ (รายได้ / สินทรัพย์เฉลี่ย)",
}

# =============================
# Helpers สำหรับคำนวณฟีเจอร์
# =============================
def add_suffix_if_needed(symbol: str, suffix: str) -> str:
    return symbol if "." in symbol else (symbol + suffix)

def _lower_index(df):
    d2 = df.copy()
    d2.index = [str(i).strip().lower() for i in d2.index]
    return d2

def _pick_at(df, keys_like, col):
    if df is None or df.empty:
        return np.nan
    df = _lower_index(df)
    idx = df.index
    for k in keys_like:
        k = k.strip().lower()
        if k in idx:
            v = df.loc[k, col] if col in df.columns else np.nan
            if pd.notna(v): return float(v)
        matches = [i for i in idx if k in i]
        for m in matches:
            v = df.loc[m, col] if col in df.columns else np.nan
            if pd.notna(v): return float(v)
    return np.nan

def _ttm_sum_partial(df, keys_like, upto_cols, min_q=2, max_q=4):
    if df is None or df.empty:
        return np.nan
    df = _lower_index(df)
    for k in keys_like:
        k = k.strip().lower()
        rows = [k] if k in df.index else [i for i in df.index if k in i]
        for row in rows:
            vals = []
            for c in upto_cols[:max_q]:
                if c in df.columns:
                    v = df.loc[row, c]
                    if pd.notna(v):
                        vals.append(float(v))
            if len(vals) >= min_q:
                return float(np.sum(vals[:max_q]))
    return np.nan

def _avg_bs(df, keys_like, col_curr, col_prev):
    a = _pick_at(df, keys_like, col_curr)
    b = _pick_at(df, keys_like, col_prev) if col_prev is not None else np.nan
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a + b) / 2.0

def safe_div(a, b):
    a = np.nan if pd.isna(a) else float(a)
    b = np.nan if pd.isna(b) else float(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def safe_div_abs(a, b):
    a = np.nan if pd.isna(a) else float(a)
    b = np.nan if pd.isna(b) else float(b)
    if pd.isna(a) or pd.isna(b) or abs(b) < 1e-12:
        return np.nan
    return a / abs(b)

@st.cache_data(show_spinner=False)
def build_one_row_for_streamlit(ticker: str, suffix: str, feature_cols: List[str]):
    sym = add_suffix_if_needed(ticker, suffix)
    tk = yf.Ticker(sym)

    fin_q = tk.quarterly_financials
    bs_q  = tk.quarterly_balance_sheet
    cf_q  = tk.quarterly_cashflow
    if fin_q is None or fin_q.empty or bs_q is None or bs_q.empty or cf_q is None or cf_q.empty:
        return pd.DataFrame()

    cols = list(fin_q.columns)  # ใหม่→เก่า
    if len(cols) == 0:
        return pd.DataFrame()

    upto_cols = cols[:4]     # 4 ไตรมาสล่าสุด
    col_curr = upto_cols[0]  # งบล่าสุด
    col_prev = upto_cols[1] if len(upto_cols) > 1 else None

    # ----- คำนวณค่า TTM -----
    rev_ttm  = _ttm_sum_partial(fin_q, ["total revenue","revenue"], upto_cols)
    cogs_ttm = _ttm_sum_partial(fin_q, ["cost of revenue","cost of goods"], upto_cols)
    gp_ttm   = rev_ttm - cogs_ttm if pd.notna(rev_ttm) and pd.notna(cogs_ttm) else np.nan
    op_ttm   = _ttm_sum_partial(fin_q, ["operating income","ebit"], upto_cols)
    ni_ttm   = _ttm_sum_partial(fin_q, ["net income"], upto_cols)
    ebit_ttm = _ttm_sum_partial(fin_q, ["ebit","operating income"], upto_cols)
    int_ttm  = _ttm_sum_partial(fin_q, ["interest expense"], upto_cols)
    ocf_ttm  = _ttm_sum_partial(cf_q, ["operating cash flow","net cash provided by operating activities"], upto_cols)

    # ----- คำนวณ revenue_growth_ttm = (TTM_now - TTM_prev) / TTM_prev -----
    rev_ttm_prev = np.nan
    if len(cols) >= 5:
        prev_cols = cols[1:5]  # เลื่อนไป 1 ไตรมาส (อีก 4 ไตรมาสถัดไป)
        rev_ttm_prev = _ttm_sum_partial(fin_q, ["total revenue","revenue"], prev_cols)
    rev_growth_ttm = safe_div(rev_ttm - rev_ttm_prev, rev_ttm_prev)

    # ----- งบแสดงฐานเฉลี่ย (ROE/ROA/Asset turnover) -----
    total_assets_avg = _avg_bs(bs_q, ["total assets"], col_curr, col_prev) if col_prev is not None else np.nan
    sh_equity_avg    = _avg_bs(bs_q, ["total stockholder equity","total shareholders equity","total equity"],
                               col_curr, col_prev) if col_prev is not None else np.nan

    total_liab_curr  = _pick_at(bs_q, ["total liab","total liabilities"], col_curr)
    curr_liab_curr   = _pick_at(bs_q, ["total current liabilities","current liabilities"], col_curr)
    sh_equity_curr   = _pick_at(bs_q, ["total stockholder equity","total shareholders equity","total equity"], col_curr)

    row = {
        "ticker": ticker,
        "asof": pd.Timestamp(col_curr).to_pydatetime(),
        "revenue_growth_ttm": rev_growth_ttm,                 # << ใช้ค่าที่คำนวณจริง
        "gross_margin_ttm": safe_div(gp_ttm, rev_ttm),
        "operating_margin_ttm": safe_div(op_ttm, rev_ttm),
        "net_margin_ttm": safe_div(ni_ttm, rev_ttm),
        "roe_ttm": safe_div(ni_ttm, sh_equity_avg),
        "roa_ttm": safe_div(ni_ttm, total_assets_avg),
        "debt_to_equity": safe_div(total_liab_curr, sh_equity_curr),
        "ocf_to_cl": safe_div(ocf_ttm, curr_liab_curr),
        "interest_coverage": safe_div_abs(ebit_ttm, int_ttm),
        "asset_turnover_ttm": safe_div(rev_ttm, total_assets_avg),
    }
    df = pd.DataFrame([row])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# =============================
# UI
# =============================
st.title("📈 สแกนหุ้น: น่าซื้อ / ไม่น่าซื้อ (Fundamental ML)")
st.caption("ใช้ข้อมูลงบการเงินจาก Yahoo Finance สร้างฟีเจอร์ 10 ตัว แล้วพยากรณ์ว่าหุ้น ‘น่าซื้อ’ หรือไม่")

with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    default_tickers = "PTT, AOT, KBANK, CPALL, ADVANC, SCB, BBL, KTB, DELTA, GULF"
    tickers_text = st.text_area("ชื่อย่อหุ้น (คั่นด้วย ,)", value=default_tickers, height=100)
    suffix = st.text_input("ส่วนขยายตลาด (Yahoo)", value=SUFFIX, help="หุ้นไทยใช้ .BK เช่น PTT.BK")
    threshold = st.slider("เกณฑ์ตัดสิน (น่าซื้อเมื่อความน่าจะเป็น ≥ เกณฑ์)", 0.1, 0.9, 0.5, 0.01)
    run_btn = st.button("🔮 พยากรณ์")

left, right = st.columns([2,1])

with left:
    st.subheader("ผลการพยากรณ์")
    if pipe is None:
        st.info("โปรดวางไฟล์ buy_model_pipeline.pkl และ model_meta.json ไว้โฟลเดอร์เดียวกับแอป")
    elif run_btn:
        tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
        rows = []
        progress = st.progress(0)
        for i, tk in enumerate(tickers, start=1):
            rows.append(build_one_row_for_streamlit(tk, suffix, FEATURE_COLS))
            progress.progress(int(i/len(tickers)*100))
        rows = [r for r in rows if not r.empty]
        if len(rows) == 0:
            st.warning("ดึงข้อมูลไม่ได้ (สัญลักษณ์ผิดหรือไม่มีงบใน Yahoo)")
        else:
            dfX = pd.concat(rows, ignore_index=True)
            X = dfX[FEATURE_COLS].astype(float)

            # ----- คำนวณ proba พร้อม normalize ให้เป็น 0..1 เสมอ -----
            try:
                proba = pipe.predict_proba(X)[:, 1].astype(float)
            except Exception:
                _patch_monotonic_cst(pipe)
                try:
                    proba = pipe.predict_proba(X)[:, 1].astype(float)
                except Exception:
                    proba = pipe.predict(X).astype(float)

            proba = np.asarray(proba, dtype=float).reshape(-1)
            while np.nanmax(proba) > 1.0:
                proba = proba / 100.0
            proba = np.clip(proba, 0.0, 1.0)
            # -------------------------------------------------------

            out = dfX[["ticker","asof"]].copy()
            out["proba_buy"] = proba
            out["pred"] = (out["proba_buy"] >= threshold).astype(int)

            df_show = out.copy()
            df_show["โอกาสน่าซื้อ (%)"] = (df_show["proba_buy"] * 100).round(1)
            df_show["ผลการประเมิน"]   = np.where(df_show["pred"] == 1, "น่าซื้อ ✅", "ไม่น่าซื้อ ❌")
            df_show = df_show.rename(columns={"ticker": "หลักทรัพย์", "asof": "งบ ณ วันที่"})[
                ["หลักทรัพย์","งบ ณ วันที่","โอกาสน่าซื้อ (%)","ผลการประเมิน"]
            ].sort_values("โอกาสน่าซื้อ (%)", ascending=False).reset_index(drop=True)

            st.dataframe(df_show, use_container_width=True, height=420)
            st.download_button(
                "💾 ดาวน์โหลด CSV",
                df_show.to_csv(index=False).encode("utf-8"),
                "predictions_streamlit_th.csv",
                "text/csv",
            )

with right:
    st.subheader("ข้อมูลโมเดล")
    st.markdown(
        f"**นิยามป้ายกำกับ (Label):** ผลตอบแทนล่วงหน้า **{FORWARD_MONTHS} เดือน > 0%** ⇒ *น่าซื้อ*  \n"
        f"**เกณฑ์ตัดสินปัจจุบัน:** proba ≥ **{threshold if 'threshold' in locals() else 0.5:.2f}**"
    )
    st.markdown("**ฟีเจอร์ที่ใช้ (10 ตัว):**")
    for key in FEATURE_COLS:
        th = FEATURE_DESC.get(key, key)
        st.markdown(f"- `{key}` — {th}")

    st.markdown(
        f"**โมเดล:** `sklearn Pipeline` (SimpleImputer → RandomForestClassifier)  \n"
        f"**เวอร์ชันรันไทม์:** scikit-learn `{sklearn.__version__}`  \n"
        f"**หมายเหตุ:** เพื่อการศึกษา **ไม่ใช่คำแนะนำการลงทุน**"
    )

    st.markdown("---")
    st.subheader("ดูกราฟราคาเร็ว ๆ")
    one = st.text_input("เลือกหุ้น (1 ตัว)", value="PTT")
    period = st.selectbox("ช่วงเวลา", ["1 ปี","2 ปี","5 ปี","ทั้งหมด"], index=1)
    period_map = {"1 ปี":"1y","2 ปี":"2y","5 ปี":"5y","ทั้งหมด":"max"}
    if st.button("📊 แสดงกราฟ"):
        sym = add_suffix_if_needed(one.strip(), suffix)
        try:
            hist = yf.Ticker(sym).history(period=period_map[period])
            if len(hist) > 0:
                st.line_chart(hist["Close"], height=250)
            else:
                st.warning("ไม่พบข้อมูลราคา")
        except Exception as e:
            st.error(str(e))

st.markdown("---")
st.caption("จัดทำเพื่อการเรียนรู้ด้าน Data Science/ML")
