"""
ML Model Manager — Loads pre-trained models from saved_models/
Training is done separately via backend/ml_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import joblib

try:
    import shap
except ImportError:
    shap = None

from backend.database import SessionLocal, CreditApplication

np.random.seed(42)

# ─── Constants ────────────────────────────────────────────────────
SVC_LIST = ["Murabaha", "Musharaka", "Ijara", "Sukuk"]
REGIONS  = [
    "Toshkent", "Samarqand", "Farg'ona", "Buxoro", "Namangan",
    "Qashqadaryo", "Andijon", "Jizzax", "Navoiy", "Sirdaryo",
    "Surxondaryo", "Xorazm", "Qoraqalpog'iston",
]
SECTORS  = ["Savdo", "Ishlab chiqarish", "Qishloq xo'jaligi", "Qurilish", "Xizmat", "Eksport"]

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

SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# Stress scenarios (for get_stress_results)
SVC_DIST = {"Murabaha": 0.42, "Musharaka": 0.22, "Ijara": 0.21, "Sukuk": 0.15}
RISK_P = {
    "Murabaha":  {"pd": 0.082, "lgd": 0.45, "vol": 0.118, "rate": (0.07, 0.18), "tenor": [6, 12, 24, 36, 60]},
    "Musharaka": {"pd": 0.148, "lgd": 0.60, "vol": 0.245, "rate": (0.12, 0.38), "tenor": [12, 24, 36, 60, 84, 120]},
    "Ijara":     {"pd": 0.062, "lgd": 0.35, "vol": 0.098, "rate": (0.05, 0.13), "tenor": [12, 24, 36, 60, 84]},
    "Sukuk":     {"pd": 0.038, "lgd": 0.28, "vol": 0.078, "rate": (0.04, 0.10), "tenor": [24, 36, 60, 84, 120]},
}


class ModelManager:
    """
    Saqlangan modellarni yuklaydi va predict, EDA, stress metodlarini ta'minlaydi.
    Model o'qitish uchun backend/ml_pipeline.py ni ishlating.
    """

    def __init__(self):
        self.best_model  = None
        self.scaler      = StandardScaler()
        self.le_svc      = LabelEncoder()
        self.le_reg      = LabelEncoder()
        self.le_sec      = LabelEncoder()
        self._trained    = False
        self.best_thr    = 0.35
        self.meta        = {}
        self.df          = pd.DataFrame()   # Runtime dataframe (EDA/stress uchun)

        # Encoder'larni oldindan to'ldirish (predict'da xato bo'lmasin uchun)
        self.le_svc.fit(SVC_LIST)
        self.le_reg.fit(REGIONS)
        self.le_sec.fit(SECTORS)

    # ─── LOAD FROM DISK ────────────────────────────────────────────
    def load_from_disk(self) -> bool:
        """
        saved_models/ dan best_model.pkl, scaler.pkl, label_encoders.pkl,
        pipeline_meta.json ni yuklaydi.

        Returns:
            True — muvaffaqiyatli yuklandi
            False — fayllar topilmadi (pipeline ishga tushirilmagan)
        """
        model_path    = os.path.join(SAVED_DIR, "best_model.pkl")
        scaler_path   = os.path.join(SAVED_DIR, "scaler.pkl")
        encoder_path  = os.path.join(SAVED_DIR, "label_encoders.pkl")
        meta_path     = os.path.join(SAVED_DIR, "pipeline_meta.json")

        missing = [p for p in [model_path, scaler_path, encoder_path] if not os.path.exists(p)]
        if missing:
            print(f"⚠️  Model fayllar topilmadi: {missing}")
            print("   Iltimos, avval pipelineni ishga tushiring:")
            print("   python -m backend.ml_pipeline")
            self._trained = False
            return False

        try:
            self.best_model = joblib.load(model_path)
            self.scaler     = joblib.load(scaler_path)
            encoders        = joblib.load(encoder_path)
            self.le_svc     = encoders["le_svc"]
            self.le_reg     = encoders["le_reg"]
            self.le_sec     = encoders["le_sec"]

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                self.best_thr = self.meta.get("best_threshold", 0.35)

            self._trained = True
            model_name    = self.meta.get("best_model_name", "Unknown")
            auc           = self.meta.get("best_auc", 0)
            trained_at    = self.meta.get("trained_at", "—")[:19]
            print(f"✅ Model yuklandi: {model_name}  AUC={auc:.4f}  ({trained_at})")
            return True

        except Exception as e:
            print(f"❌ Model yuklashda xato: {e}")
            self._trained = False
            return False

    def _ensure_loaded(self):
        """Predict'dan oldin model yuklanganligini tekshiradi."""
        if not self._trained:
            ok = self.load_from_disk()
            if not ok:
                raise RuntimeError(
                    "Model o'qitilmagan. Avval pipelineni ishga tushiring: "
                    "python -m backend.ml_pipeline"
                )

    def _build_runtime_df(self):
        """
        EDA / stress metodlari uchun DataFrame kerak.
        DB dan yuklab, kamida 50 ta bo'lsa uni ishlatadi. Bo'lmasa o'qitish data'siga fallback qiladi.
        """
        if self.df is not None and len(self.df) >= 50:
            return

        try:
            from backend.database import engine
            df_db = pd.read_sql("SELECT * FROM credit_applications", engine)
            if len(df_db) >= 50:
                print(f"📊 DB data used for runtime ({len(df_db)} rows)")
                if self._trained:
                    self.df = self.score_df(df_db)
                else:
                    self.df = df_db
                return
        except Exception as e:
            print(f"⚠️  DB load error: {e}")

        # Fallback: Pipeline artifactlari yoki o'qitish ma'lumotlari
        try:
            from backend.ml_pipeline import IslomiyBankPipeline
            pipe = IslomiyBankPipeline()
            # O'qitishda ishlatilgan CSV bo'lsa o'shani, bo'lmasa sintetikni yuklaydi
            pipe.load_data()
            if self._trained:
                print("📊 Training data used for runtime stats (fallback)")
                self.df = self.score_df(pipe.df)
            else:
                self.df = pipe.df
        except Exception as e:
            print(f"❌ Fallback error: {e}")
            self.df = pd.DataFrame()

    def score_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame dagi barcha qatorlar uchun risk ko'rsatkichlarini hisoblaydi."""
        if not self._trained or df is None or len(df) == 0:
            return df
        
        df = df.copy()
        try:
            # 1. Feature Engineering
            df["xizmat_enc"] = self.le_svc.transform(df["xizmat_turi"].fillna("Murabaha").apply(lambda x: x if x in self.le_svc.classes_ else "Murabaha"))
            df["mintaqa_enc"] = self.le_reg.transform(df["mintaqa"].fillna("Toshkent").apply(lambda x: x if x in self.le_reg.classes_ else "Toshkent"))
            df["sektor_enc"] = self.le_sec.transform(df["sektor"].fillna("Savdo").apply(lambda x: x if x in self.le_sec.classes_ else "Savdo"))

            df["credit_ltv"]        = df["kredit_ball"] / (df["ltv_nisbati"].replace(0, 1e-6))
            df["sharia_risk_proxy"] = (1 - df["sharia_audit"]) * (df.get("gharar_darajasi", 0) + df.get("maysir_ekspozitsiya", 0) + 1)
            df["risk_adj_ret"]      = df["foyda_stavkasi"] / (df.get("bozor_volatilligi", 0.12).replace(0, 1e-6))
            df["gharar_maysir"]     = df.get("gharar_darajasi", 0) + df.get("maysir_ekspozitsiya", 0)
            df["dsr_ltv"]           = df["qarz_xizmat_nisbati"] * df["ltv_nisbati"]
            df["macro_stress"]      = df.get("inflyatsiya", 0.1) + df.get("valyuta_tebranishi", 0.05) - df.get("yim_osishi", 0.05)
            df["prev_default_rate"] = df["oldingi_defaultlar"] / (df["oldingi_kreditlar"].replace(0, 1e-6))
            
            if "sharia_score" not in df.columns:
                df["sharia_score"] = df.get("sharia_audit", 0.95)

            # 2. Prediction
            X = df[FEATURES].fillna(0)
            X_sc = self.scaler.transform(X)
            probs = self.best_model.predict_proba(X_sc)[:, 1]
            df["pd_qiymati"] = probs

            # 3. Metrikalar
            df["risk_darajasi"] = df["pd_qiymati"].apply(
                lambda p: "Past" if p < 0.10 else ("O'rta" if p < 0.25 else ("Yuqori" if p < 0.45 else "Juda Yuqori"))
            )
            
            # Simplified LGD/EAD for batch
            df["ead"] = df["moliyalash_miqdori"] * (1 + df["foyda_stavkasi"] * (df["muddat_oy"]/12) * 0.5)
            df["lgd"] = 0.45 # default
            df["kutilgan_zarar"] = df["pd_qiymati"] * df["ead"] * df["lgd"]
            
            # SRI
            W = {"pd": 0.35, "mkt": 0.25, "sh": 0.25, "liq": 0.15}
            sh_r = (1 - df["sharia_audit"]) + df.get("gharar_darajasi", 0) * 0.5
            lq_r = 1 - df["likvidlik"]
            df["sri_indeksi"] = W["pd"] * df["pd_qiymati"] + W["mkt"] * df.get("bozor_volatilligi", 0.12) + W["sh"] * sh_r + W["liq"] * lq_r

            return df
        except Exception as e:
            print(f"❌ Batch scoring error: {e}")
            return df

    # ─── PREDICT ───────────────────────────────────────────────────
    def predict(self, inp: dict) -> dict:
        """Bitta mijoz uchun risk bashorati."""
        self._ensure_loaded()

        row = inp.copy()

        # sharia_score yo'q bo'lsa
        if not row.get("sharia_score"):
            row["sharia_score"] = row.get("sharia_audit", 0.95)

        # Label encoding
        try:
            row["xizmat_enc"] = int(self.le_svc.transform([row["xizmat_turi"]])[0])
        except Exception:
            row["xizmat_enc"] = 0
        try:
            row["mintaqa_enc"] = int(self.le_reg.transform([row["mintaqa"]])[0])
        except Exception:
            row["mintaqa_enc"] = 0
        try:
            row["sektor_enc"] = int(self.le_sec.transform([row["sektor"]])[0])
        except Exception:
            row["sektor_enc"] = 0

        # Feature engineering
        row["credit_ltv"]        = row["kredit_ball"] / (row["ltv_nisbati"] + 1e-6)
        row["sharia_risk_proxy"] = (1 - row["sharia_audit"]) * (row["gharar_darajasi"] + row["maysir_ekspozitsiya"] + 1)
        row["risk_adj_ret"]      = row["foyda_stavkasi"] / (row["bozor_volatilligi"] + 1e-6)
        row["gharar_maysir"]     = row["gharar_darajasi"] + row["maysir_ekspozitsiya"]
        row["dsr_ltv"]           = row["qarz_xizmat_nisbati"] * row["ltv_nisbati"]
        row["macro_stress"]      = row["inflyatsiya"] + row["valyuta_tebranishi"] - row["yim_osishi"]
        row["prev_default_rate"] = row["oldingi_defaultlar"] / (row["oldingi_kreditlar"] + 1e-6)

        X_input = np.array([[row.get(f, 0.0) for f in FEATURES]])
        X_scaled = self.scaler.transform(X_input)

        # Prediction
        ens_p = float(self.best_model.predict_proba(X_scaled)[:, 1][0])

        # SHAP
        shap_explain = None
        if shap:
            try:
                explainer = shap.TreeExplainer(self.best_model)
                sv = explainer.shap_values(X_scaled)
                if isinstance(sv, list):
                    vals = sv[1][0]
                else:
                    vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
                top_idx = np.argsort(np.abs(vals))[-3:][::-1]
                shap_explain = {FEATURES[i]: float(vals[i]) for i in top_idx}
            except Exception as e:
                print(f"SHAP Error: {e}")

        # Risk darajasi
        if ens_p < 0.10:   rlvl, rcode = "Past", 0
        elif ens_p < 0.25: rlvl, rcode = "O'rta", 1
        elif ens_p < 0.45: rlvl, rcode = "Yuqori", 2
        else:              rlvl, rcode = "Juda Yuqori", 3

        # Financial calcs
        ead = row["moliyalash_miqdori"] * (1 + row["foyda_stavkasi"] * row["muddat_oy"] / 12 * 0.5)
        lgd_est = {"Murabaha": 0.45, "Musharaka": 0.60, "Ijara": 0.35, "Sukuk": 0.28}.get(row["xizmat_turi"], 0.45)
        kutilgan_zarar = ens_p * ead * lgd_est

        mu  = row["foyda_stavkasi"] / 252
        sig = row["bozor_volatilligi"] / np.sqrt(252)
        sim = np.random.normal(mu, sig, 10000)
        var95  = float(np.percentile(sim, 5))
        cvar95 = float(sim[sim <= var95].mean())
        ex     = sim - 0.13 / 252
        sharpe = float(np.sqrt(252) * ex.mean() / ex.std()) if ex.std() > 0 else 0.0

        # SRI
        sh_r = ((1 - row.get("sharia_score", 0.95))
                + row["gharar_darajasi"] * 0.3
                + (1 - row.get("zakat_status", 1)) * 0.1
                + row.get("partnership_risk", 0) * 0.05)
        mk_r = row["bozor_volatilligi"]
        lq_r = 1 - row["likvidlik"]
        sri  = 0.30 * ens_p + 0.20 * mk_r + 0.35 * sh_r + 0.15 * lq_r
        sri_grade = "A" if sri < 0.08 else ("B" if sri < 0.15 else ("C" if sri < 0.25 else "D"))

        # Tavsiya
        tavsiyalar = [
            "Kredit tasdiqlash tavsiya etiladi. Shariat muvofiqligi yuqori.",
            "Qo'shimcha garov yoki shartlar bilan tasdiqlash mumkin.",
            "Ehtiyotkorlik bilan ko'rib chiqing — yuqori moliyaviy/shariat riski!",
            "Kredit berishdan bosh tortish tavsiya etiladi.",
        ]
        tavsiya = tavsiyalar[rcode]

        # Save to DB
        db = SessionLocal()
        try:
            app_record = CreditApplication(
                xizmat_turi=row["xizmat_turi"], mintaqa=row["mintaqa"], sektor=row["sektor"],
                kredit_ball=float(row["kredit_ball"]), yosh=int(row.get("yosh", 35)),
                tajriba=int(row.get("tajriba", 5)),
                oldingi_kreditlar=int(row.get("oldingi_kreditlar", 0)),
                oldingi_defaultlar=int(row.get("oldingi_defaultlar", 0)),
                moliyalash_miqdori=float(row["moliyalash_miqdori"]),
                muddat_oy=int(row.get("muddat_oy", 24)),
                foyda_stavkasi=float(row.get("foyda_stavkasi", 0.1)),
                ltv_nisbati=float(row.get("ltv_nisbati", 0.7)),
                qarz_xizmat_nisbati=float(row.get("qarz_xizmat_nisbati", 0.3)),
                likvidlik=float(row.get("likvidlik", 0.8)),
                leverage=float(row.get("leverage", 0.4)),
                garov_sifati=int(row.get("garov_sifati", 3)),
                sharia_audit=float(row.get("sharia_audit", 0.9)),
                sharia_score=float(row.get("sharia_score", 0.95)),
                zakat_status=int(row.get("zakat_status", 1)),
                partnership_risk=int(row.get("partnership_risk", 0)),
                gharar_darajasi=float(row.get("gharar_darajasi", 0.05)),
                maysir_ekspozitsiya=float(row.get("maysir_ekspozitsiya", 0.02)),
                halal_sertifikat=int(row.get("halal_sertifikat", 1)),
                bozor_volatilligi=float(row.get("bozor_volatilligi", 0.1)),
                yim_osishi=float(row.get("yim_osishi", 0.05)),
                inflyatsiya=float(row.get("inflyatsiya", 0.1)),
                valyuta_tebranishi=float(row.get("valyuta_tebranishi", 0.05)),
                neft_narxi=float(row.get("neft_narxi", 0.03)),
                bank_indeksi=float(row.get("bank_indeksi", 0.04)),
                pd_qiymati=float(ens_p),
                default_holati=int(row.get("default_holati", 0)),
                risk_darajasi=rlvl, sri_indeksi=float(sri),
            )
            db.add(app_record)
            db.commit()
        except Exception as db_err:
            print(f"DB Save Error: {db_err}")
            db.rollback()
        finally:
            db.close()

        return {
            "pd_qiymati":            round(ens_p, 4),
            "risk_darajasi":          rlvl,
            "risk_kodi":              rcode,
            "default_ehtimoli_pct":   round(ens_p * 100, 2),
            "ead":                    round(ead),
            "lgd":                    round(lgd_est, 4),
            "kutilgan_zarar":         round(kutilgan_zarar),
            "var_95":                 round(var95, 6),
            "cvar_95":                round(cvar95, 6),
            "sharpe_ratio":           round(sharpe, 3),
            "sri_indeksi":            round(sri, 5),
            "sri_daraja":             sri_grade,
            "tavsiya":                tavsiya,
            "model_probabilities":    {"best_model": round(ens_p, 4)},
            "shap_explain":           shap_explain,
        }

    # ─── EDA STATS ─────────────────────────────────────────────────
    def get_eda_stats(self) -> dict:
        self._ensure_loaded()
        self._build_runtime_df()
        df = self.df
        result = {
            "total_records":  len(df),
            "default_rate":   round(df["default_holati"].mean() * 100, 2) if "default_holati" in df.columns else 0.0,
            "avg_kredit_ball": round(df["kredit_ball"].mean(), 1) if "kredit_ball" in df.columns else 0.0,
            "avg_pd":         round(df["pd_qiymati"].mean() * 100, 2) if "pd_qiymati" in df.columns else 0.0,
            "avg_ltv":        round(df["ltv_nisbati"].mean(), 4) if "ltv_nisbati" in df.columns else 0.0,
            "avg_sharia":     round(df["sharia_audit"].mean(), 4) if "sharia_audit" in df.columns else 0.0,
            "model_info":     {
                "name":       self.meta.get("best_model_name", "Unknown"),
                "auc":        self.meta.get("best_auc", 0),
                "threshold":  self.best_thr,
                "trained_at": self.meta.get("trained_at", "—")[:19],
                "data_source": self.meta.get("data_source", "—"),
                "n_samples":  self.meta.get("n_samples", 0),
            },
        }
        if "risk_darajasi" in df.columns:
            result["risk_distribution"] = df["risk_darajasi"].value_counts().to_dict()
        if "xizmat_turi" in df.columns and "default_holati" in df.columns:
            result["by_service"] = df.groupby("xizmat_turi")["default_holati"].agg(["count", "mean"]).round(4).to_dict()
        if "sektor" in df.columns and "default_holati" in df.columns:
            result["sector_default"] = df.groupby("sektor")["default_holati"].mean().round(4).to_dict()
        if "mintaqa" in df.columns and "default_holati" in df.columns:
            result["region_default"] = df.groupby("mintaqa")["default_holati"].mean().round(4).to_dict()
        return result

    # ─── MODEL PERFORMANCE ─────────────────────────────────────────
    def get_model_performance(self) -> dict:
        self._ensure_loaded()
        all_aucs = self.meta.get("all_model_aucs", {})
        result = {}
        for name, auc in all_aucs.items():
            result[name] = {
                "auc":      round(auc, 4),
                "f1":       0.0,
                "accuracy": 0.0,
                "roc_fpr":  [],
                "roc_tpr":  [],
                "confusion_matrix": [],
            }
        # Add Ensemble entry prominently
        result["Ensemble"] = {
            "auc":      self.meta.get("ensemble_auc", 0),
            "f1":       0.0,
            "accuracy": 0.0,
            "roc_fpr":  [],
            "roc_tpr":  [],
            "confusion_matrix": [],
        }
        return result

    # ─── STRESS TEST ───────────────────────────────────────────────
    def get_stress_results(self) -> dict:
        self._ensure_loaded()
        self._build_runtime_df()
        df = self.df
        STRESS = {
            "Asosiy":           {"pm": 1.0, "rd": 0.000},
            "Yengil Stress":    {"pm": 1.5, "rd": 0.020},
            "O'rta Stress":     {"pm": 2.5, "rd": 0.050},
            "Og'ir Stress":     {"pm": 4.0, "rd": 0.100},
            "COVID-19":         {"pm": 5.5, "rd": 0.080},
            "Valyuta":          {"pm": 3.0, "rd": 0.060},
        }
        result = {}
        for sc, par in STRESS.items():
            row = {"total": 0}
            for svc in SVC_LIST:
                sub = df[df["xizmat_turi"] == svc] if "xizmat_turi" in df.columns else df
                if len(sub) == 0:
                    row[svc] = 0.0; continue
                spd = np.clip(sub["pd_qiymati"].values * par["pm"], 0, 1) if "pd_qiymati" in sub.columns else np.full(len(sub), 0.1)
                sea = (sub["ead"].values if "ead" in sub.columns else sub.get("moliyalash_miqdori", pd.Series()).values) * (1 + par["rd"])
                lgd = sub["lgd"].values if "lgd" in sub.columns else np.full(len(sub), 0.45)
                el  = (spd * sea * lgd).sum() / 1e6
                row[svc] = round(el, 2)
                row["total"] += el
            row["total"] = round(row["total"], 2)
            result[sc] = row
        return result

    # ─── RISK METRICS ──────────────────────────────────────────────
    def get_risk_metrics(self) -> dict:
        self._ensure_loaded()
        self._build_runtime_df()
        df = self.df
        W  = {"pd": 0.35, "mkt": 0.25, "sh": 0.25, "liq": 0.15}
        result = {}
        for svc in SVC_LIST:
            sub = df[df["xizmat_turi"] == svc] if "xizmat_turi" in df.columns else df
            if len(sub) == 0:
                continue
            # mu/sig calculation with NaNs handling
            mu_val = sub["foyda_stavkasi"].mean() if "foyda_stavkasi" in sub.columns else 0.10
            if pd.isna(mu_val): mu_val = 0.10
            
            sig_val = sub["bozor_volatilligi"].mean() if "bozor_volatilligi" in sub.columns else 0.12
            if pd.isna(sig_val) or sig_val <= 0: sig_val = 0.12
            
            mu = mu_val / 252
            sig = sig_val / np.sqrt(252)
            
            sim = np.random.normal(mu, sig, 10000)
            
            var95_val = np.percentile(sim, 5)
            var95 = float(var95_val) if not np.isnan(var95_val) else -0.01
            
            cvar_subset = sim[sim <= var95]
            if len(cvar_subset) > 0:
                cvar95 = float(cvar_subset.mean())
            else:
                cvar95 = var95 * 1.2

            ex = sim - 0.13 / 252
            ex_std = ex.std()
            sharpe = float(np.sqrt(252) * ex.mean() / ex_std) if ex_std > 0 else 0.0
            if np.isnan(sharpe): sharpe = 0.0

            pd_r = sub["pd_qiymati"].mean() if "pd_qiymati" in sub.columns else 0.1
            if pd.isna(pd_r): pd_r = 0.1
            
            mk_r = sub["bozor_volatilligi"].mean() if "bozor_volatilligi" in sub.columns else 0.12
            if pd.isna(mk_r): mk_r = 0.12
            
            sh_r = ((1 - sub["sharia_audit"].mean()) + sub.get("gharar_darajasi", 0).mean() * 0.5) if "sharia_audit" in sub.columns else 0.1
            if pd.isna(sh_r): sh_r = 0.1
            
            lq_r = 1 - sub["likvidlik"].mean() if "likvidlik" in sub.columns else 0.2
            if pd.isna(lq_r): lq_r = 0.2
            
            sri  = W["pd"] * pd_r + W["mkt"] * mk_r + W["sh"] * sh_r + W["liq"] * lq_r
            if pd.isna(sri): sri = 0.15
            
            sri_grade = "A" if sri < 0.08 else ("B" if sri < 0.14 else ("C" if sri < 0.22 else "D"))

            ead_col = sub["ead"] if "ead" in sub.columns else sub.get("moliyalash_miqdori", pd.Series(np.ones(len(sub))))
            lgd_col = sub["lgd"] if "lgd" in sub.columns else pd.Series(np.full(len(sub), 0.45))
            pd_col  = sub["pd_qiymati"] if "pd_qiymati" in sub.columns else pd.Series(np.full(len(sub), pd_r))
            
            EL_sum = (pd_col * ead_col * lgd_col).sum()
            EL     = float(EL_sum) / 1e6 if not pd.isna(EL_sum) else 0.0
            
            loss = pd_col * ead_col * lgd_col
            loss_std = loss.std()
            UL     = float(loss_std) * 2.33 / 1e6 if not pd.isna(loss_std) else 0.0

            result[svc] = {
                "var_95":       round(float(var95), 6),
                "cvar_95":      round(float(cvar95), 6),
                "sharpe":       round(float(sharpe), 3),
                "sri":          round(float(sri), 5),
                "sri_grade":    str(sri_grade),
                "el_mln":       round(float(EL), 2),
                "ul_mln":       round(float(UL), 2),
                "avg_pd":       round(float(pd_r) * 100, 2),
                "default_count": int(sub["default_holati"].sum()) if "default_holati" in sub.columns else 0,
                "total_count":  int(len(sub)),
            }
        return result

    # ─── INGEST NEW DATA (online learning support) ─────────────────
    def ingest_new_data(self, new_df: pd.DataFrame) -> dict:
        """Yangi ma'lumotlarni DB ga qo'shadi (dublikat tekshiruvi bilan)."""
        import hashlib
        import time

        print(f"[{time.strftime('%H:%M:%S')}] Yangi ma'lumotlar: {len(new_df)} qator")

        # Target column normalization
        for cand in ["default", "is_default", "risk", "label", "status", "target"]:
            if cand in new_df.columns and "default_holati" not in new_df.columns:
                new_df = new_df.rename(columns={cand: "default_holati"})
                break

        db = SessionLocal()
        new_count = dup_count = 0
        try:
            for _, row in new_df.iterrows():
                r_dict = row.to_dict()
                relevant = "|".join([str(r_dict.get(f, "")) for f in FEATURES])
                r_hash   = hashlib.sha256(relevant.encode()).hexdigest()
                if db.query(CreditApplication).filter(CreditApplication.data_hash == r_hash).first():
                    dup_count += 1; continue
                record = CreditApplication(
                    xizmat_turi=str(r_dict.get("xizmat_turi", "Murabaha")),
                    mintaqa=str(r_dict.get("mintaqa", "Toshkent")),
                    sektor=str(r_dict.get("sektor", "Savdo")),
                    kredit_ball=float(r_dict.get("kredit_ball", 650)),
                    yosh=int(r_dict.get("yosh", 35)),
                    tajriba=int(r_dict.get("tajriba", 5)),
                    oldingi_kreditlar=int(r_dict.get("oldingi_kreditlar", 0)),
                    oldingi_defaultlar=int(r_dict.get("oldingi_defaultlar", 0)),
                    moliyalash_miqdori=float(r_dict.get("moliyalash_miqdori", 1000000)),
                    muddat_oy=int(r_dict.get("muddat_oy", 24)),
                    foyda_stavkasi=float(r_dict.get("foyda_stavkasi", 0.1)),
                    ltv_nisbati=float(r_dict.get("ltv_nisbati", 0.7)),
                    qarz_xizmat_nisbati=float(r_dict.get("qarz_xizmat_nisbati", 0.3)),
                    likvidlik=float(r_dict.get("likvidlik", 0.8)),
                    leverage=float(r_dict.get("leverage", 0.4)),
                    garov_sifati=int(r_dict.get("garov_sifati", 3)),
                    sharia_audit=float(r_dict.get("sharia_audit", 0.9)),
                    sharia_score=float(r_dict.get("sharia_score", 0.95)),
                    zakat_status=int(r_dict.get("zakat_status", 1)),
                    partnership_risk=int(r_dict.get("partnership_risk", 0)),
                    gharar_darajasi=float(r_dict.get("gharar_darajasi", 0.05)),
                    maysir_ekspozitsiya=float(r_dict.get("maysir_ekspozitsiya", 0.02)),
                    halal_sertifikat=int(r_dict.get("halal_sertifikat", 1)),
                    bozor_volatilligi=float(r_dict.get("bozor_volatilligi", 0.1)),
                    yim_osishi=float(r_dict.get("yim_osishi", 0.05)),
                    inflyatsiya=float(r_dict.get("inflyatsiya", 0.1)),
                    valyuta_tebranishi=float(r_dict.get("valyuta_tebranishi", 0.05)),
                    neft_narxi=float(r_dict.get("neft_narxi", 0.03)),
                    bank_indeksi=float(r_dict.get("bank_indeksi", 0.04)),
                    pd_qiymati=0.0, default_holati=int(r_dict.get("default_holati", 0)),
                    risk_darajasi="Unknown", sri_indeksi=0.0, data_hash=r_hash,
                )
                db.add(record); new_count += 1
            db.commit()
        except Exception as e:
            db.rollback(); print(f"DB Error: {e}")
        finally:
            db.close()

        print(f"[{time.strftime('%H:%M:%S')}] {new_count} yangi, {dup_count} dublikat")
        return {"new": new_count, "duplicates": dup_count}


# Singleton
manager = ModelManager()
