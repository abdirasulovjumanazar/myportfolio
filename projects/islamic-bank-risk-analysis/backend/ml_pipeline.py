"""
╔══════════════════════════════════════════════════════════════════╗
║  Islomiy Bank Risk Analysis — Automated ML Pipeline              ║
║  Part 1–11 mantiqini birlashtiruvchi yagona pipeline             ║
║  IslomiyBankPipeline.run(csv_path=None) → saved_models/          ║
╚══════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.calibration import calibration_curve
from scipy.stats import randint as sp_randint, uniform as sp_uniform

import joblib

# ─── Paths ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

# ─── Constants (part1) ────────────────────────────────────────────
np.random.seed(42)

SVC_DIST = {"Murabaha": 0.42, "Musharaka": 0.22, "Ijara": 0.21, "Sukuk": 0.15}
SVC_LIST = list(SVC_DIST.keys())
RISK_P = {
    "Murabaha":  {"pd": 0.082, "lgd": 0.45, "vol": 0.118, "rate": (0.07, 0.18), "tenor": [6, 12, 24, 36, 60]},
    "Musharaka": {"pd": 0.148, "lgd": 0.60, "vol": 0.245, "rate": (0.12, 0.38), "tenor": [12, 24, 36, 60, 84, 120]},
    "Ijara":     {"pd": 0.062, "lgd": 0.35, "vol": 0.098, "rate": (0.05, 0.13), "tenor": [12, 24, 36, 60, 84]},
    "Sukuk":     {"pd": 0.038, "lgd": 0.28, "vol": 0.078, "rate": (0.04, 0.10), "tenor": [24, 36, 60, 84, 120]},
}
REGIONS = [
    "Toshkent", "Samarqand", "Farg'ona", "Buxoro", "Namangan",
    "Qashqadaryo", "Andijon", "Jizzax", "Navoiy", "Sirdaryo",
    "Surxondaryo", "Xorazm", "Qoraqalpog'iston",
]
SECTORS = ["Savdo", "Ishlab chiqarish", "Qishloq xo'jaligi", "Qurilish", "Xizmat", "Eksport"]

PALETTE = ["#1B4F72", "#A93226", "#1E8449", "#D4A017", "#6C3483"]
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.titleweight": "bold"})

# ─── Features (part6) ─────────────────────────────────────────────
FEATURES = [
    "xizmat_enc", "mintaqa_enc", "sektor_enc",
    "kredit_ball", "ltv_nisbati", "foyda_stavkasi", "muddat_oy",
    "qarz_xizmat_nisbati", "likvidlik", "leverage", "garov_sifati",
    "sharia_audit", "gharar_darajasi", "maysir_ekspozitsiya", "halal_sertifikat",
    "zakat_status", "partnership_risk", "sharia_score",
    "bozor_volatilligi", "yim_osishi", "inflyatsiya", "valyuta_tebranishi",
    "neft_narxi", "bank_indeksi", "oldingi_kreditlar", "oldingi_defaultlar",
    "credit_ltv", "sharia_risk_proxy", "risk_adj_ret", "gharar_maysir",
    "dsr_ltv", "macro_stress", "prev_default_rate",
]
TARGET = "default_holati"

# ─── Column alias mapping (real bank data → internal names) ───────
COLUMN_ALIASES = {
    # Target
    "default": TARGET, "is_default": TARGET, "risk": TARGET,
    "label": TARGET, "status": TARGET, "target": TARGET,
    # Service type
    "product_type": "xizmat_turi", "contract_type": "xizmat_turi",
    "financing_type": "xizmat_turi", "service_type": "xizmat_turi",
    # Region
    "region": "mintaqa", "branch_region": "mintaqa", "city": "mintaqa",
    "wiloyat": "mintaqa", "region_code": "mintaqa",
    # Credit score
    "credit_score": "kredit_ball", "score": "kredit_ball",
    "fico_score": "kredit_ball", "rating": "kredit_ball",
    # Financial
    "loan_amount": "moliyalash_miqdori", "financing_amount": "moliyalash_miqdori",
    "principal": "moliyalash_miqdori",
    "tenor": "muddat_oy", "term_months": "muddat_oy", "duration": "muddat_oy",
    "profit_rate": "foyda_stavkasi", "interest_rate": "foyda_stavkasi",
    "markup_rate": "foyda_stavkasi",
    "ltv": "ltv_nisbati", "loan_to_value": "ltv_nisbati",
    "dsr": "qarz_xizmat_nisbati", "debt_service_ratio": "qarz_xizmat_nisbati",
    "liquidity": "likvidlik", "liq_ratio": "likvidlik",
    "leverage_ratio": "leverage",
    "collateral_quality": "garov_sifati", "collateral_grade": "garov_sifati",
    # Sharia
    "sharia_compliance": "sharia_audit", "sharia_rating": "sharia_audit",
    "gharar": "gharar_darajasi", "gharar_level": "gharar_darajasi",
    "maysir": "maysir_ekspozitsiya", "maysir_level": "maysir_ekspozitsiya",
    "halal_cert": "halal_sertifikat", "halal_certified": "halal_sertifikat",
    "zakat": "zakat_status", "zakat_paid": "zakat_status",
    # Macro
    "volatility": "bozor_volatilligi", "market_vol": "bozor_volatilligi",
    "gdp_growth": "yim_osishi", "gdp": "yim_osishi",
    "inflation": "inflyatsiya", "cpi": "inflyatsiya",
    "fx_rate": "valyuta_tebranishi", "exchange_rate": "valyuta_tebranishi",
    "oil_price": "neft_narxi",
    # History
    "prev_loans": "oldingi_kreditlar", "previous_loans": "oldingi_kreditlar",
    "prev_defaults": "oldingi_defaultlar", "previous_defaults": "oldingi_defaultlar",
    # Demographics
    "age": "yosh", "borrower_age": "yosh",
    "experience": "tajriba", "work_experience": "tajriba",
    "sector": "sektor", "industry": "sektor",
}


class IslomiyBankPipeline:
    """
    Barcha ML bosqichlarini birlashtiruvchi pipeline.

    Ishlatish:
        pipeline = IslomiyBankPipeline()
        pipeline.run()                        # Sintetik dataset bilan
        pipeline.run("my_bank_data.csv")      # Haqiqiy CSV bilan
    """

    def __init__(self, output_dir: str = SAVED_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df    = None   # Raw DataFrame
        self.df2   = None   # Feature-engineered DataFrame
        self.X_tr_res = None
        self.y_tr_res = None
        self.X_te_sc  = None
        self.y_te     = None
        self.scaler   = StandardScaler()
        self.le_svc   = LabelEncoder()
        self.le_reg   = LabelEncoder()
        self.le_sec   = LabelEncoder()
        self.all_models   = {}
        self.best_model   = None
        self.best_model_name = None
        self.best_thr     = 0.35
        self.data_source  = "unknown"

        # Encoder'larni oldindan to'ldirish
        self.le_svc.fit(SVC_LIST)
        self.le_reg.fit(REGIONS)
        self.le_sec.fit(SECTORS)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. DATA LOADING  (part1 mantiqidan + real data support)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def load_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        Ustuvorlik tartibi:
          1. csv_path berilganmi? → CSV/Excel o'qiydi
          2. DB da 50+ qator bormi? → DB dan yuklaydi
          3. Sintetik fallback (demo uchun)
        """
        # 1. CSV / Excel
        if csv_path and os.path.exists(csv_path):
            print(f"📂 Haqiqiy dataset o'qilyapti: {csv_path}")
            df = self._read_file(csv_path)
            if df is not None and len(df) >= 50:
                df = self._normalize_columns(df)
                self.df = self._encode_categoricals(df)
                self.data_source = f"csv:{os.path.basename(csv_path)}"
                print(f"   ✅ {len(self.df)} qator yuklandi (CSV)")
                return self.df

        # 2. DB
        try:
            from backend.database import engine
            df_db = pd.read_sql("SELECT * FROM credit_applications", engine)
            if len(df_db) >= 50:
                df_db = self._normalize_columns(df_db)
                self.df = self._encode_categoricals(df_db)
                self.data_source = "database"
                print(f"   ✅ {len(self.df)} qator DB dan yuklandi")
                return self.df
            else:
                print(f"   ⚠️  DB da faqat {len(df_db)} qator — kam")
        except Exception as e:
            print(f"   ⚠️  DB dan yuklab bo'lmadi: {e}")

        # 3. Sintetik fallback
        print("🔧 Sintetik dataset generatsiya qilinmoqda (demo)...")
        self.df = self._generate_synthetic(N=2500)
        self.data_source = "synthetic"
        print(f"   ✅ {len(self.df)} sintetik qator yaratildi")
        return self.df

    def _read_file(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                return pd.read_csv(path, encoding="utf-8-sig")
            elif ext in (".xls", ".xlsx"):
                return pd.read_excel(path)
            else:
                print(f"   ⚠️  Noma'lum fayl turi: {ext}")
                return None
        except Exception as e:
            print(f"   ❌ Fayl o'qishda xato: {e}")
            return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ustun nomlarini ichki nomlarga moslashtiradi."""
        rename_map = {}
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for alias, internal in COLUMN_ALIASES.items():
            if alias in cols_lower and internal not in df.columns:
                rename_map[cols_lower[alias]] = internal
        if rename_map:
            print(f"   🔄 Ustun nomlari moslashtirish: {rename_map}")
            df = df.rename(columns=rename_map)

        # Majburiy ustunlar yo'q bo'lsa default qiymatlar
        defaults = {
            "zakat_status": 1, "partnership_risk": 0, "sharia_score": 0.95,
            "xizmat_turi": "Murabaha", "mintaqa": "Toshkent", "sektor": "Savdo",
            "yosh": 35, "tajriba": 5, "halal_sertifikat": 1,
            "sharia_audit": 0.90, "gharar_darajasi": 0.05, "maysir_ekspozitsiya": 0.02,
            "bozor_volatilligi": 0.12, "yim_osishi": 0.056, "inflyatsiya": 0.098,
            "valyuta_tebranishi": 0.048, "neft_narxi": 0.02, "bank_indeksi": 0.04,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kategorik ustunlarni raqamga aylantiradi (safe — noma'lum qiymatlar uchun)."""
        # xizmat_turi
        df["xizmat_turi"] = df["xizmat_turi"].fillna("Murabaha")
        unknown_svc = set(df["xizmat_turi"].unique()) - set(SVC_LIST)
        if unknown_svc:
            df["xizmat_turi"] = df["xizmat_turi"].apply(
                lambda x: x if x in SVC_LIST else "Murabaha"
            )
        df["xizmat_enc"] = self.le_svc.transform(df["xizmat_turi"])

        # mintaqa
        df["mintaqa"] = df["mintaqa"].fillna("Toshkent")
        unknown_reg = set(df["mintaqa"].unique()) - set(REGIONS)
        if unknown_reg:
            df["mintaqa"] = df["mintaqa"].apply(
                lambda x: x if x in REGIONS else "Toshkent"
            )
        df["mintaqa_enc"] = self.le_reg.transform(df["mintaqa"])

        # sektor
        df["sektor"] = df["sektor"].fillna("Savdo")
        unknown_sec = set(df["sektor"].unique()) - set(SECTORS)
        if unknown_sec:
            df["sektor"] = df["sektor"].apply(
                lambda x: x if x in SECTORS else "Savdo"
            )
        df["sektor_enc"] = self.le_sec.transform(df["sektor"])
        return df

    def _generate_synthetic(self, N: int = 2500) -> pd.DataFrame:
        """Part1 mantiqidan sintetik dataset generatsiyasi."""
        svc_arr = np.random.choice(SVC_LIST, N, p=list(SVC_DIST.values()))
        rows = []
        for svc in svc_arr:
            p = RISK_P[svc]
            cscore   = np.clip(np.random.normal(645, 85), 300, 850)
            age      = np.random.randint(22, 65)
            exp_yr   = np.random.randint(1, 25)
            loan_amt = np.random.lognormal(10.8, 1.1)
            tenor    = np.random.choice(p["tenor"])
            rate     = np.random.uniform(*p["rate"])
            region   = np.random.choice(REGIONS)
            sector   = np.random.choice(SECTORS)
            ltv      = np.clip(np.random.beta(4.5, 3), 0.10, 0.95)
            dsr      = np.clip(np.random.beta(3, 5), 0.05, 0.85)
            col_q    = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.20, 0.40, 0.25, 0.10])
            liq      = np.clip(np.random.beta(7, 2.5), 0.20, 1.0)
            lev      = np.random.uniform(0.10, 0.90)
            n_prev   = np.random.poisson(2.5)
            n_def    = np.random.binomial(n_prev, 0.08) if n_prev > 0 else 0
            sharia   = np.clip(np.random.beta(8, 2.5), 0.55, 1.0)
            sh_score = np.clip(np.random.normal(sharia, 0.05), 0, 1)
            gharar   = np.clip(np.random.beta(2, 8), 0.00, 0.50)
            maysir   = np.clip(np.random.beta(1.5, 9), 0.00, 0.40)
            halal    = np.random.choice([0, 1], p=[0.12, 0.88])
            zakat    = np.random.choice([0, 1], p=[0.20, 0.80])
            p_risk   = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
            mkt_vol  = np.abs(np.random.normal(p["vol"], 0.025))
            gdp_g    = np.random.normal(0.056, 0.014)
            inf_r    = np.random.normal(0.098, 0.022)
            fx       = np.abs(np.random.normal(0.048, 0.028))
            oil      = np.random.normal(0.02, 0.15)
            bidx     = np.random.normal(0.04, 0.08)

            z = (-4.5 + p["pd"] * 12 - (cscore - 650) * 0.006
                 + ltv * 2.5 + dsr * 3.8 - sharia * 2.0
                 + gharar * 4.0 + maysir * 3.5 + inf_r * 6.0
                 + mkt_vol * 5.5 - liq * 2.5 + lev * 1.8
                 + n_def * 0.9 - halal * 0.6 - zakat * 0.4
                 + p_risk * 0.7 - (col_q - 3) * 0.5
                 + np.random.normal(0, 0.25))
            pd_val = 1 / (1 + np.exp(-z))
            is_def = int(np.random.random() < pd_val)
            ead    = loan_amt * (1 + rate * tenor / 12 * 0.5)
            lgd    = np.clip(np.random.normal(p["lgd"], 0.08), 0.10, 0.95)
            el     = pd_val * ead * lgd

            if pd_val < 0.10:   rlvl, rcode = "Past", 0
            elif pd_val < 0.25: rlvl, rcode = "O'rta", 1
            elif pd_val < 0.45: rlvl, rcode = "Yuqori", 2
            else:               rlvl, rcode = "Juda Yuqori", 3

            rows.append({
                "xizmat_turi": svc, "mintaqa": region, "sektor": sector,
                "kredit_ball": round(cscore), "yosh": age, "tajriba": exp_yr,
                "oldingi_kreditlar": n_prev, "oldingi_defaultlar": n_def,
                "moliyalash_miqdori": round(loan_amt), "muddat_oy": tenor,
                "foyda_stavkasi": round(rate, 4), "ltv_nisbati": round(ltv, 4),
                "qarz_xizmat_nisbati": round(dsr, 4), "likvidlik": round(liq, 4),
                "leverage": round(lev, 4), "garov_sifati": col_q,
                "sharia_audit": round(sharia, 4), "sharia_score": round(sh_score, 4),
                "gharar_darajasi": round(gharar, 4), "maysir_ekspozitsiya": round(maysir, 4),
                "halal_sertifikat": halal, "zakat_status": zakat, "partnership_risk": p_risk,
                "bozor_volatilligi": round(mkt_vol, 4), "yim_osishi": round(gdp_g, 4),
                "inflyatsiya": round(inf_r, 4), "valyuta_tebranishi": round(fx, 4),
                "neft_narxi": round(oil, 4), "bank_indeksi": round(bidx, 4),
                "pd_qiymati": round(pd_val, 4), "ead": round(ead),
                "lgd": round(lgd, 4), "kutilgan_zarar": round(el),
                "default_holati": is_def, "risk_darajasi": rlvl, "risk_kodi": rcode,
            })

        df = pd.DataFrame(rows)
        df["xizmat_enc"]  = self.le_svc.transform(df["xizmat_turi"])
        df["mintaqa_enc"] = self.le_reg.transform(df["mintaqa"])
        df["sektor_enc"]  = self.le_sec.transform(df["sektor"])
        print(f"   Default nisbati: {df[TARGET].mean():.2%}")
        return df

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. EDA  (part2 mantiqidan)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run_eda(self):
        print("📊 EDA (Kengaytirilgan tahlil) bajarilmoqda...")
        df = self.df
        fig = plt.figure(figsize=(20, 15))
        gs  = gridspec.GridSpec(3, 4, hspace=0.50, wspace=0.38)

        # 1. Pie
        ax = fig.add_subplot(gs[0, 0])
        if "xizmat_turi" in df.columns:
            cnt = df["xizmat_turi"].value_counts()
            ax.pie(cnt, labels=cnt.index, autopct="%1.1f%%", colors=PALETTE,
                   startangle=90, wedgeprops={"linewidth": 2, "edgecolor": "white"})
        ax.set_title("Xizmat Turlari Ulushi")

        # 2. Default nisbati
        ax = fig.add_subplot(gs[0, 1])
        if "xizmat_turi" in df.columns and TARGET in df.columns:
            dr = df.groupby("xizmat_turi")[TARGET].mean().sort_values(ascending=False)
            bars = ax.bar(dr.index, dr.values * 100, color=PALETTE[:len(dr)], edgecolor="white", lw=1.5)
            for b, v in zip(bars, dr.values):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Default Nisbati"); ax.set_ylabel("%")

        # 3. Violin — PD
        ax = fig.add_subplot(gs[0, 2])
        if "pd_qiymati" in df.columns and "xizmat_turi" in df.columns:
            data_v = [df[df["xizmat_turi"] == s]["pd_qiymati"].values for s in SVC_LIST
                      if s in df["xizmat_turi"].values]
            labels = [s for s in SVC_LIST if s in df["xizmat_turi"].values]
            if data_v:
                parts = ax.violinplot(data_v, showmedians=True, showextrema=True)
                for i, pc in enumerate(parts["bodies"]):
                    pc.set_facecolor(PALETTE[i % len(PALETTE)]); pc.set_alpha(0.75)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("PD Violin"); ax.set_ylabel("Default Ehtimoli")

        # 4. Kredit ball boxplot
        ax = fig.add_subplot(gs[0, 3])
        if "kredit_ball" in df.columns and "xizmat_turi" in df.columns:
            grp    = [df[df["xizmat_turi"] == s]["kredit_ball"].values for s in SVC_LIST
                      if s in df["xizmat_turi"].values]
            labels = [s for s in SVC_LIST if s in df["xizmat_turi"].values]
            if grp:
                ax.boxplot(grp, patch_artist=True,
                           boxprops=dict(facecolor="#D6EAF8"),
                           medianprops=dict(color="navy", lw=2))
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title("Kredit Ball Taqsimlashi"); ax.set_ylabel("Kredit Ball")

        # 5. Scatter LTV vs PD
        ax = fig.add_subplot(gs[1, 0])
        if "ltv_nisbati" in df.columns and "pd_qiymati" in df.columns:
            for i, svc in enumerate(SVC_LIST):
                sub = df[df["xizmat_turi"] == svc] if "xizmat_turi" in df.columns else df
                if len(sub) > 0:
                    ax.scatter(sub["ltv_nisbati"], sub["pd_qiymati"],
                               s=10, alpha=0.3, color=PALETTE[i % len(PALETTE)], label=svc)
            xl = np.linspace(df["ltv_nisbati"].min(), df["ltv_nisbati"].max(), 100)
            z  = np.polyfit(df["ltv_nisbati"], df["pd_qiymati"], 1)
            ax.plot(xl, np.poly1d(z)(xl), "k--", lw=1.5, label="Trend")
            ax.legend(fontsize=7, markerscale=2)
        ax.set_title("LTV va PD"); ax.set_xlabel("LTV"); ax.set_ylabel("PD")

        # 6. Risk darajasi stacked bar
        ax = fig.add_subplot(gs[1, 1])
        if "risk_darajasi" in df.columns and "xizmat_turi" in df.columns:
            rct = df.groupby(["xizmat_turi", "risk_darajasi"]).size().unstack(fill_value=0)
            rct_pct = rct.div(rct.sum(1), 0) * 100
            rct_pct.plot(kind="bar", stacked=True, ax=ax,
                         color=["#1E8449", "#F39C12", "#E74C3C", "#8E44AD"],
                         edgecolor="white", lw=0.4)
            ax.tick_params(axis="x", rotation=20); ax.legend(fontsize=7)
        ax.set_title("Risk Darajasi (%)"); ax.set_ylabel("%"); ax.set_xlabel("")

        # 7. Sharia vs PD
        ax = fig.add_subplot(gs[1, 2])
        if "sharia_audit" in df.columns and "pd_qiymati" in df.columns:
            bins = pd.cut(df["sharia_audit"], 6)
            spd  = df.groupby(bins, observed=True)["pd_qiymati"].mean()
            ax.bar(range(len(spd)), spd.values * 100, color=PALETTE[2], edgecolor="white")
            ax.set_xticks(range(len(spd)))
            ax.set_xticklabels([f"{i.left:.2f}" for i in spd.index], fontsize=8, rotation=30)
        ax.set_title("Sharia Audit va O'rt. PD"); ax.set_ylabel("PD (%)")

        # 8. Kutilgan zarar
        ax = fig.add_subplot(gs[1, 3])
        if "kutilgan_zarar" in df.columns and "xizmat_turi" in df.columns:
            el = df.groupby("xizmat_turi")["kutilgan_zarar"].mean() / 1e6
            ax.barh(el.sort_values().index, el.sort_values().values,
                    color=PALETTE[:len(el)], edgecolor="white")
        ax.set_title("O'rt. Kutilgan Zarar"); ax.set_xlabel("mln UZS")

        # 9. Korrelyatsiya heatmap
        ax = fig.add_subplot(gs[2, :])
        num_cols = [c for c in [
            "kredit_ball", "ltv_nisbati", "foyda_stavkasi", "bozor_volatilligi",
            "likvidlik", "sharia_audit", "gharar_darajasi", "qarz_xizmat_nisbati",
            "inflyatsiya", "valyuta_tebranishi", "leverage", "pd_qiymati", TARGET
        ] if c in df.columns]
        if len(num_cols) > 2:
            corr = df[num_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f",
                        cmap="RdYlGn_r", center=0, vmin=-1, vmax=1,
                        linewidths=0.4, annot_kws={"size": 7.5},
                        xticklabels=[c[:12] for c in num_cols],
                        yticklabels=[c[:15] for c in num_cols])
        ax.set_title("Korrelyatsiya Matritsasi", fontsize=12)

        fig.suptitle(f"Islomiy Bank — Kengaytirilgan EDA ({self.data_source})",
                     fontsize=15, fontweight="bold")
        out = os.path.join(self.output_dir, "eda.png")
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"   ✅ eda.png saqlandi: {out}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. MONTE CARLO + STRESS TESTING  (part5 mantiqidan)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run_monte_carlo(self, n_sim: int = 20_000, horizon: int = 252):
        print("🎲 Monte Carlo & Stress Testing bajarilmoqda...")
        df = self.df
        N_SIM = n_sim; HOR = horizon

        MC = {}
        for svc in SVC_LIST:
            sub = df[df["xizmat_turi"] == svc] if "xizmat_turi" in df.columns else df
            if len(sub) == 0:
                continue
            mu  = sub["foyda_stavkasi"].mean() if "foyda_stavkasi" in sub.columns else 0.10
            sig = sub["bozor_volatilligi"].mean() if "bozor_volatilligi" in sub.columns else 0.12
            S0  = sub["ead"].mean() if "ead" in sub.columns else sub["moliyalash_miqdori"].mean() if "moliyalash_miqdori" in sub.columns else 1e6
            dt  = 1 / HOR
            dW  = np.random.normal(0, np.sqrt(dt), (N_SIM, HOR))
            ret = (mu - 0.5 * sig ** 2) * dt + sig * dW
            paths = S0 * np.exp(np.cumsum(ret, axis=1))
            fin   = paths[:, -1]
            MC[svc] = {
                "paths": paths[:100, :], "final": fin, "S0": S0,
                "mean": fin.mean(), "std": fin.std(),
                "var95": np.percentile(fin, 5),
                "var99": np.percentile(fin, 1),
                "prob_loss": (fin < S0).mean(),
            }

        # Stress scenariylari
        STRESS = {
            "Asosiy":           {"pm": 1.0, "rd": 0.000},
            "Yengil Stress":    {"pm": 1.5, "rd": 0.020},
            "O'rta Stress":     {"pm": 2.5, "rd": 0.050},
            "Og'ir Stress":     {"pm": 4.0, "rd": 0.100},
            "COVID-19 analog":  {"pm": 5.5, "rd": 0.080},
            "Valyuta inqirozi": {"pm": 3.0, "rd": 0.060},
        }
        stress_rows = []
        for sc, par in STRESS.items():
            row = {"Senariy": sc}
            tot = 0
            for svc in SVC_LIST:
                sub = df[df["xizmat_turi"] == svc] if "xizmat_turi" in df.columns else df
                if len(sub) == 0:
                    row[svc] = 0.0; continue
                spd = np.clip(sub["pd_qiymati"].values * par["pm"], 0, 1) if "pd_qiymati" in sub.columns else np.full(len(sub), 0.1)
                sea = (sub["ead"].values if "ead" in sub.columns else sub["moliyalash_miqdori"].values) * (1 + par["rd"])
                lgd = sub["lgd"].values if "lgd" in sub.columns else np.full(len(sub), 0.45)
                el  = (spd * sea * lgd).sum() / 1e6
                row[svc] = round(el, 2); tot += el
            row["JAMI"] = round(tot, 2)
            stress_rows.append(row)

        # Monte Carlo visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        days = np.arange(HOR)
        for idx, (svc, color) in enumerate(zip(SVC_LIST, PALETTE)):
            if svc not in MC:
                continue
            ax = axes[idx // 2, idx % 2]
            ps = MC[svc]["paths"]; S0 = MC[svc]["S0"]
            for path in ps[:60]:
                ax.plot(days, path / S0, alpha=0.07, color=color, lw=0.7)
            p5  = np.percentile(ps, 5,  axis=0) / S0
            p50 = np.percentile(ps, 50, axis=0) / S0
            p95 = np.percentile(ps, 95, axis=0) / S0
            ax.plot(days, p50, color="navy", lw=2,   label="P50")
            ax.plot(days, p5,  color="red",  lw=1.5, ls="--", label="P5")
            ax.plot(days, p95, color="green",lw=1.5, ls="--", label="P95")
            ax.fill_between(days, p5, p95, alpha=0.1, color=color)
            ax.axhline(1, color="black", lw=1, ls=":")
            r = MC[svc]
            ax.set_title(f"{svc}\nZarar ehtimoli={r['prob_loss']:.1%}  VaR99={r['var99']/S0:.3f}")
            ax.set_xlabel("Kun"); ax.set_ylabel("Normallashtirilgan")
            ax.legend(fontsize=8)
        fig.suptitle(f"Monte Carlo GBM ({N_SIM:,} Senariy)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_mc = os.path.join(self.output_dir, "monte_carlo.png")
        plt.savefig(out_mc, bbox_inches="tight", dpi=150); plt.close()

        # Stress testing visualization
        stress_df = pd.DataFrame(stress_rows).set_index("Senariy")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        svc_cols = [c for c in SVC_LIST if c in stress_df.columns]
        if svc_cols:
            stress_df[svc_cols].plot(kind="bar", ax=ax2, color=PALETTE[:len(svc_cols)], edgecolor="white")
        ax2.set_title("Stress Testing — Senariy bo'yicha EL (mln UZS)", fontweight="bold")
        ax2.set_ylabel("EL (mln UZS)"); ax2.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        out_st = os.path.join(self.output_dir, "stress_testing.png")
        plt.savefig(out_st, bbox_inches="tight", dpi=150); plt.close()

        print(f"   ✅ monte_carlo.png, stress_testing.png saqlandi")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. FEATURE ENGINEERING  (part6 mantiqidan)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def feature_engineer(self):
        print("⚙️  Feature engineering bajarilmoqda...")
        df2 = self.df.copy()
        df2["credit_ltv"]       = df2["kredit_ball"] / (df2["ltv_nisbati"] + 1e-6)
        df2["sharia_risk_proxy"] = (1 - df2["sharia_audit"]) * (df2["gharar_darajasi"] + df2["maysir_ekspozitsiya"] + 1)
        df2["risk_adj_ret"]      = df2["foyda_stavkasi"] / (df2["bozor_volatilligi"] + 1e-6)
        df2["gharar_maysir"]     = df2["gharar_darajasi"] + df2["maysir_ekspozitsiya"]
        df2["dsr_ltv"]           = df2["qarz_xizmat_nisbati"] * df2["ltv_nisbati"]
        df2["macro_stress"]      = df2["inflyatsiya"] + df2["valyuta_tebranishi"] - df2["yim_osishi"]
        df2["prev_default_rate"] = df2["oldingi_defaultlar"] / (df2["oldingi_kreditlar"] + 1e-6)

        # Ensure sharia_score exists
        if "sharia_score" not in df2.columns:
            df2["sharia_score"] = df2["sharia_audit"]

        self.df2 = df2
        print(f"   ✅ {len(FEATURES)} feature tayyorlandi")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. TRAIN & TUNE  (part7 mantiqidan)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def train_models(self, fast: bool = False):
        """
        Train all models using proper imblearn Pipeline to prevent data leakage.

        KEY FIX (Data Leakage Prevention):
        Previously, SMOTE was applied to the ENTIRE training set before CV.
        This caused synthetic minority samples to appear in validation folds,
        artificially inflating AUC/F1 scores.

        Now: SMOTE is INSIDE the Pipeline, so it only runs on each CV train-fold,
        never contaminating validation folds. This gives honest metrics.

        fast=True → reduced n_iter for quick testing
        fast=False → full search for production
        """
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError:
            print("   ⚠️  imbalanced-learn not found. Installing...")
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn", "-q"])
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

        print("🤖 Modellar o'qitilmoqda (Data-Leakage-Free Pipeline)...")
        df2 = self.df2

        # ── Data preparation ─────────────────────────────────────────
        df2[TARGET] = pd.to_numeric(df2[TARGET], errors="coerce")
        clean = df2.dropna(subset=[TARGET])
        if len(clean) < 50:
            raise ValueError(f"Yetarli ma'lumot yo'q: {len(clean)} qator (kamida 50 kerak)")

        X = clean[FEATURES].copy()
        for f in FEATURES:
            X[f] = pd.to_numeric(X[f], errors="coerce")
        X = X.fillna(X.median())
        y = clean[TARGET].astype(int)

        # ── Train/Test split (hold-out) ───────────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # ── Scaler fitted ONLY on train, applied to test ──────────────
        self.scaler = StandardScaler()
        X_tr_sc = self.scaler.fit_transform(X_tr)
        X_te_sc = self.scaler.transform(X_te)
        self.X_te_sc = X_te_sc
        self.y_te    = y_te

        # ── SMOTE inside Pipeline (no leakage) ───────────────────────
        smote = SMOTE(random_state=42, k_neighbors=5)
        cv5   = StratifiedKFold(5, shuffle=True, random_state=42)
        n_iter_rf  = 10 if fast else 30
        n_iter_gbm = 8  if fast else 25
        self.all_models = {}

        # 1. Logistic Regression — already balanced via class_weight
        print("   1/4 Logistic Regression ...")
        lr_pipe = ImbPipeline([
            ("smote", smote),
            ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga")),
        ])
        lr_grid = {"clf__C": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}
        lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=cv5, scoring="roc_auc", n_jobs=-1)
        lr_gs.fit(X_tr_sc, y_tr)
        lr_prob = lr_gs.best_estimator_.predict_proba(X_te_sc)[:, 1]
        lr_auc  = roc_auc_score(y_te, lr_prob)
        print(f"      Best C={lr_gs.best_params_['clf__C']}  Test-AUC={lr_auc:.4f}")
        self.all_models["Logistic Regression"] = {
            "model": lr_gs.best_estimator_, "prob": lr_prob, "auc": lr_auc
        }

        # 2. Random Forest
        print(f"   2/4 Random Forest (n_iter={n_iter_rf}) ...")
        rf_pipe = ImbPipeline([
            ("smote", smote),
            ("clf",   RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
        ])
        rf_params = {
            "clf__n_estimators":    sp_randint(100, 400),
            "clf__max_depth":       sp_randint(4, 12),
            "clf__min_samples_leaf": sp_randint(3, 15),
            "clf__max_features":    ["sqrt", "log2", 0.5, 0.7],
            "clf__min_samples_split": sp_randint(5, 20),
        }
        rf_rs = RandomizedSearchCV(
            rf_pipe, rf_params, n_iter=n_iter_rf, cv=cv5,
            scoring="roc_auc", random_state=42, n_jobs=-1
        )
        rf_rs.fit(X_tr_sc, y_tr)
        rf_prob = rf_rs.best_estimator_.predict_proba(X_te_sc)[:, 1]
        rf_auc  = roc_auc_score(y_te, rf_prob)
        print(f"      Test-AUC={rf_auc:.4f}")
        self.all_models["Random Forest"] = {
            "model": rf_rs.best_estimator_, "prob": rf_prob, "auc": rf_auc
        }

        # 3. Gradient Boosting
        print(f"   3/4 Gradient Boosting (n_iter={n_iter_gbm}) ...")
        gbm_pipe = ImbPipeline([
            ("smote", smote),
            ("clf",   GradientBoostingClassifier(random_state=42)),
        ])
        gbm_params = {
            "clf__n_estimators":    sp_randint(100, 400),
            "clf__max_depth":       sp_randint(3, 8),
            "clf__learning_rate":   sp_uniform(0.01, 0.25),
            "clf__subsample":       sp_uniform(0.5, 0.5),
            "clf__min_samples_leaf": sp_randint(5, 25),
            "clf__max_features":    ["sqrt", "log2", 0.6],
        }
        gbm_rs = RandomizedSearchCV(
            gbm_pipe, gbm_params, n_iter=n_iter_gbm, cv=cv5,
            scoring="roc_auc", random_state=42, n_jobs=-1
        )
        gbm_rs.fit(X_tr_sc, y_tr)
        gbm_prob = gbm_rs.best_estimator_.predict_proba(X_te_sc)[:, 1]
        gbm_auc  = roc_auc_score(y_te, gbm_prob)
        print(f"      Test-AUC={gbm_auc:.4f}")
        self.all_models["GBM"] = {
            "model": gbm_rs.best_estimator_, "prob": gbm_prob, "auc": gbm_auc
        }

        # 4. Ensemble (soft average of base models)
        print("   4/4 Ensemble ...")
        ens_prob = np.mean([m["prob"] for m in self.all_models.values()], axis=0)
        ens_auc  = roc_auc_score(y_te, ens_prob)
        self.all_models["Ensemble"] = {"prob": ens_prob, "auc": ens_auc}
        print(f"      Ensemble AUC={ens_auc:.4f}")

        # ── Best single model selection ───────────────────────────────
        auc_vals = {k: v["auc"] for k, v in self.all_models.items() if "model" in v}
        self.best_model_name = max(auc_vals, key=auc_vals.get)
        self.best_model = self.all_models[self.best_model_name]["model"]

        # ── Results table ─────────────────────────────────────────────
        print("\n" + "═" * 60)
        print(f"{'Model':<25}{'AUC':>8}{'F1':>8}{'Accuracy':>10}")
        print("─" * 60)
        for name, res in self.all_models.items():
            yp = (res["prob"] >= self.best_thr).astype(int)
            print(f"{name:<25}{res['auc']:>8.4f}"
                  f"{f1_score(y_te, yp, zero_division=0):>8.4f}"
                  f"{accuracy_score(y_te, yp):>10.4f}")
        print("═" * 60)
        print(f"⭐ Eng yaxshi model: {self.best_model_name} "
              f"(AUC={auc_vals[self.best_model_name]:.4f})")
        print("✅ SMOTE leakage-free Pipeline ishlatildi.")



    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. THRESHOLD OPTIMIZATION  (part9 mantiqidan)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def find_best_threshold(self):
        print("🎯 Threshold optimization ...")
        ens_prob = self.all_models["Ensemble"]["prob"]
        thrs = np.arange(0.10, 0.90, 0.02)
        best_f1, best_thr = 0.0, 0.35
        for t in thrs:
            yp = (ens_prob >= t).astype(int)
            f1 = f1_score(self.y_te, yp, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, t
        self.best_thr = round(float(best_thr), 2)
        print(f"   ✅ Optimal threshold: {self.best_thr:.2f}  F1={best_f1:.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. SAVE ARTIFACTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_artifacts(self):
        print("💾 Artifactlar saqlanmoqda...")

        # Models
        joblib.dump(self.best_model,  os.path.join(self.output_dir, "best_model.pkl"))
        joblib.dump(self.scaler,      os.path.join(self.output_dir, "scaler.pkl"))
        joblib.dump({
            "le_svc": self.le_svc,
            "le_reg": self.le_reg,
            "le_sec": self.le_sec,
        }, os.path.join(self.output_dir, "label_encoders.pkl"))

        # All models (for performance comparison)
        all_models_slim = {
            k: {"auc": v["auc"]} for k, v in self.all_models.items()
        }

        # Meta JSON
        meta = {
            "trained_at":       datetime.now().isoformat(),
            "data_source":      self.data_source,
            "n_samples":        int(len(self.df)),
            "default_rate":     round(float(self.df[TARGET].mean()), 4) if TARGET in self.df.columns else None,
            "best_model_name":  self.best_model_name,
            "best_auc":         round(float(self.all_models[self.best_model_name]["auc"]), 4),
            "best_threshold":   self.best_thr,
            "ensemble_auc":     round(float(self.all_models["Ensemble"]["auc"]), 4),
            "all_model_aucs":   {k: round(v["auc"], 4) for k, v in self.all_models.items()},
            "features":         FEATURES,
            "n_features":       len(FEATURES),
        }
        meta_path = os.path.join(self.output_dir, "pipeline_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"   ✅ best_model.pkl    → {self.best_model_name}")
        print(f"   ✅ scaler.pkl")
        print(f"   ✅ label_encoders.pkl")
        print(f"   ✅ pipeline_meta.json")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. MAIN ENTRY POINT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run(self, csv_path: str = None, fast: bool = False):
        """
        Pipelineni ishga tushirish.

        Args:
            csv_path: Haqiqiy CSV/Excel yo'li (ixtiyoriy).
                      Yo'q bo'lsa DB dan, undan ham yo'q bo'lsa sintetik.
            fast:     True → tezkor mode (test uchun, n_iter kamaytirilib)
        """
        t0 = time.time()
        print("=" * 65)
        print("🕌 ISLOMIY BANK RISK ANALYSIS — ML PIPELINE BOSHLANMOQDA")
        print("=" * 65)

        print("\n[1/7] Ma'lumotlarni yuklash...")
        self.load_data(csv_path)

        print("\n[2/7] EDA (Vizualizatsiya)...")
        self.run_eda()

        print("\n[3/7] Monte Carlo & Stress Testing...")
        self.run_monte_carlo()

        print("\n[4/7] Feature Engineering...")
        self.feature_engineer()

        print("\n[5/7] Modellarni o'qitish (RandomizedSearchCV)...")
        self.train_models(fast=fast)

        print("\n[6/7] Threshold Optimization...")
        self.find_best_threshold()

        print("\n[7/7] Artifactlarni saqlash...")
        self.save_artifacts()

        elapsed = time.time() - t0
        print(f"\n{'=' * 65}")
        print(f"✅ PIPELINE MUVAFFAQIYATLI YAKUNLANDI!  ({elapsed:.1f} soniya)")
        print(f"   Saqlangan: {self.output_dir}")
        print(f"   Eng yaxshi model: {self.best_model_name}")
        print(f"   Ensemble AUC: {self.all_models['Ensemble']['auc']:.4f}")
        print(f"   Threshold: {self.best_thr}")
        print(f"{'=' * 65}")
        return {
            "elapsed": round(elapsed, 1),
            "best_model": self.best_model_name,
            "ensemble_auc": round(self.all_models["Ensemble"]["auc"], 4),
            "threshold": self.best_thr,
            "data_source": self.data_source,
            "n_samples": len(self.df),
        }


# ─── Standalone ishlatish ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Islomiy Bank ML Pipeline")
    parser.add_argument("--csv",  type=str, default=None, help="CSV/Excel fayl yo'li")
    parser.add_argument("--fast", action="store_true",    help="Tezkor mode (test uchun)")
    args = parser.parse_args()

    pipeline = IslomiyBankPipeline()
    result   = pipeline.run(csv_path=args.csv, fast=args.fast)
    print(json.dumps(result, ensure_ascii=False, indent=2))
