"""
🕌 Islomiy Bank Risk Tahlili — FastAPI Backend
"""
import sys, os
try:
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, Exception):
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import asyncio
from sqlalchemy.orm import Session

from backend.schemas import CreditInput, RiskOutput
from backend.models import manager
from backend.auto_analyzer import run_auto_analysis
from backend.database import init_db, SessionLocal, CreditApplication, User, get_db
from backend.auth import (
    authenticate_user, create_access_token, get_current_user, RoleChecker,
    get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES, verify_password
)
from backend.integration import router as integration_router
from fastapi import Response, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

# ─── Pipeline status (retrain progress tracking) ──────────────────
_pipeline_status = {
    "running": False,
    "last_result": None,
    "last_error": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: DB init + modelni yuklayman (.pkl dan)"""
    print("🕌 Islomiy Bank Risk API ishga tushmoqda...")
    init_db()
    
    # Create default admin if not exists
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("👤 Creating default admin user...")
            new_admin = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                role="ADMIN"
            )
            db.add(new_admin)
            db.commit()
    finally:
        db.close()

    ok = manager.load_from_disk()
    if not ok:
        print("⚠️  Model yuklanmadi. /api/retrain endpoint orqali o'qiting.")
    print("✅ API tayyor!")
    yield

app = FastAPI(
    title="🕌 Islomiy Bank Risk API",
    description=(
        "Islomiy banklarda moliyaviy xizmatlar risklarini ML asosida baholash.\n\n"
        "## Bank integratsiyasi\n"
        "Bank tizimlari (1C, Oracle FLEXCUBE, IBSO) uchun `/api/v1/integration/score` endpointi mavjud.\n"
        "Autentifikatsiya: `X-API-Key` header."
    ),
    version="4.1",
    lifespan=lifespan,
    contact={"name": "Islom Bank Risk Team", "email": "admin@islamicbank.uz"},
    license_info={"name": "Commercial License"},
)

# ─── Integration Router ────────────────────────────────────────────
app.include_router(integration_router)

# ─── Environment flags ─────────────────────────────────────────────
IS_PRODUCTION = os.getenv("PRODUCTION", "false").lower() == "true"

# ─── CORS ─────────────────────────────────────────────────────────
# In production, set ALLOWED_ORIGINS env var: "https://yourdomain.com,https://www..."
_default_origins = [
    "http://localhost:8080", "http://localhost:8000",
    "http://localhost:3000", "http://localhost:5500",
    "http://127.0.0.1:8080", "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
]
_env_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Set-Cookie"],
)

# ─── Auth Endpoints ───────────────────────────────────────────────
@app.post("/api/auth/login")
async def login(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Authenticate user and set secure HttpOnly session cookie."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login yoki parol xato",
        )

    access_token = create_access_token(data={"sub": user.username})

    # secure=True only on HTTPS (production). Controlled by PRODUCTION env var.
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=IS_PRODUCTION,
    )
    return {"status": "success", "username": user.username, "role": user.role}

@app.post("/api/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "logged_out"}

@app.get("/api/auth/me")
async def get_me(user: User = Depends(get_current_user)):
    return {
        "username": user.username,
        "role": user.role,
        "created_at": user.created_at
    }

# ─── Frontend static files ─────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# ─────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "🕌 Islomiy Bank Risk API v4.0", "docs": "/docs"}


@app.get("/api/health")
async def health():
    # Agar model hali yuklanmagan bo'lsayu, pipeline tugagan bo'lsa (fayllar bor bo'lsa)
    # avtomatik yuklashga urinib ko'ramiz.
    if not manager._trained:
        manager.load_from_disk()
        
    return {
        "status": "ok",
        "model_trained": manager._trained,
        "model_name": manager.meta.get("best_model_name", "—"),
        "auc": manager.meta.get("best_auc", 0),
    }


@app.post("/api/predict")
def predict_risk(inp: CreditInput, user: User = Depends(RoleChecker(["ADMIN", "ANALYST"]))):
    """Kredit risk bashorati"""
    try:
        result = manager.predict(inp.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eda-stats")
def eda_stats(user: User = Depends(RoleChecker(["ADMIN", "ANALYST", "VIEWER"]))):
    """EDA statistikasi"""
    try:
        return manager.get_eda_stats()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk-metrics")
def risk_metrics(user: User = Depends(RoleChecker(["ADMIN", "ANALYST", "VIEWER"]))):
    """VaR, CVaR, SRI, Basel EL/UL metrikalari"""
    try:
        return manager.get_risk_metrics()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stress-test")
def stress_test(user: User = Depends(RoleChecker(["ADMIN", "ANALYST", "VIEWER"]))):
    """Stress test scenariylari"""
    try:
        return manager.get_stress_results()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-performance")
def model_performance(user: User = Depends(RoleChecker(["ADMIN", "ANALYST", "VIEWER"]))):
    """ML model metrikalari va ROC ma'lumotlari"""
    try:
        return manager.get_model_performance()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monte-carlo/{service}")
def monte_carlo(service: str, n_sim: int = 5000, horizon: int = 252, user: User = Depends(RoleChecker(["ADMIN", "ANALYST"]))):
    """Monte Carlo GBM simulatsiyasi"""
    try:
        manager._ensure_loaded()
        manager._build_runtime_df()
        df = manager.df
        if service not in ["Murabaha", "Musharaka", "Ijara", "Sukuk"]:
            raise HTTPException(status_code=400, detail="Noto'g'ri xizmat turi")
        sub = df[df["xizmat_turi"] == service] if "xizmat_turi" in df.columns else df
        mu  = sub["foyda_stavkasi"].mean() if "foyda_stavkasi" in sub.columns else 0.10
        sig = sub["bozor_volatilligi"].mean() if "bozor_volatilligi" in sub.columns else 0.12
        S0  = sub["ead"].mean() if "ead" in sub.columns else sub["moliyalash_miqdori"].mean() if "moliyalash_miqdori" in sub.columns else 1e6
        dt  = 1 / horizon
        dW  = np.random.normal(0, np.sqrt(dt), (n_sim, horizon))
        ret = (mu - 0.5 * sig ** 2) * dt + sig * dW
        paths = S0 * np.exp(np.cumsum(ret, axis=1))
        fin  = paths[:, -1]
        p5   = (np.percentile(paths, 5,  axis=0) / S0).tolist()
        p50  = (np.percentile(paths, 50, axis=0) / S0).tolist()
        p95  = (np.percentile(paths, 95, axis=0) / S0).tolist()
        return {
            "service":    service,
            "S0":         round(S0),
            "n_sim":      n_sim,
            "horizon":    horizon,
            "p5":         [round(x, 4) for x in p5[::5]],
            "p50":        [round(x, 4) for x in p50[::5]],
            "p95":        [round(x, 4) for x in p95[::5]],
            "days":       list(range(0, horizon, 5)),
            "var99":      round(float(np.percentile(fin, 1)) / S0, 4),
            "prob_loss":  round(float((fin < S0).mean()), 4),
            "mean_final": round(float(fin.mean()) / S0, 4),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
def portfolio(n_portfolios: int = 5000, user: User = Depends(RoleChecker(["ADMIN", "ANALYST"]))):
    """Portfel optimizatsiyasi — Efficient Frontier"""
    try:
        manager._ensure_loaded()
        manager._build_runtime_df()
        df = manager.df
        SVC = ["Murabaha", "Musharaka", "Ijara", "Sukuk"]
        mu_v  = np.array([df[df["xizmat_turi"] == s]["foyda_stavkasi"].mean() if "xizmat_turi" in df.columns else 0.1 for s in SVC])
        sig_v = np.array([df[df["xizmat_turi"] == s]["bozor_volatilligi"].mean() if "xizmat_turi" in df.columns else 0.12 for s in SVC])
        pd_v  = np.array([df[df["xizmat_turi"] == s]["pd_qiymati"].mean() if "pd_qiymati" in df.columns else 0.1 for s in SVC])
        adj_mu = mu_v * (1 - pd_v)
        corr_m = np.array([[1.00, 0.35, 0.28, 0.15], [0.35, 1.00, 0.42, 0.22],
                            [0.28, 0.42, 1.00, 0.30], [0.15, 0.22, 0.30, 1.00]])
        cov_m = np.outer(sig_v, sig_v) * corr_m
        RF_RATE = 0.13
        p_ret, p_vol, p_sr, p_w = [], [], [], []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(4))
            r = np.dot(w, adj_mu)
            v = np.sqrt(w @ cov_m @ w)
            s = (r - RF_RATE) / v if v > 0 else 0
            p_ret.append(r); p_vol.append(v); p_sr.append(s); p_w.append(w.tolist())
        p_ret = np.array(p_ret); p_vol = np.array(p_vol); p_sr = np.array(p_sr)
        msr_i = int(np.argmax(p_sr)); mvr_i = int(np.argmin(p_vol))
        cur_w = np.array([0.42, 0.22, 0.21, 0.15])
        return {
            "scatter_vol": [round(x * 100, 3) for x in p_vol[::10].tolist()],
            "scatter_ret": [round(x * 100, 3) for x in p_ret[::10].tolist()],
            "scatter_sr":  [round(x, 3) for x in p_sr[::10].tolist()],
            "max_sharpe":  {
                "weights": dict(zip(SVC, [round(x, 4) for x in p_w[msr_i]])),
                "return":  round(float(p_ret[msr_i]) * 100, 2),
                "vol":     round(float(p_vol[msr_i]) * 100, 2),
                "sharpe":  round(float(p_sr[msr_i]), 3),
            },
            "min_risk":    {
                "weights": dict(zip(SVC, [round(x, 4) for x in p_w[mvr_i]])),
                "return":  round(float(p_ret[mvr_i]) * 100, 2),
                "vol":     round(float(p_vol[mvr_i]) * 100, 2),
            },
            "current":     {
                "weights": dict(zip(SVC, cur_w.tolist())),
                "return":  round(float(np.dot(cur_w, adj_mu)) * 100, 2),
                "vol":     round(float(np.sqrt(cur_w @ cov_m @ cur_w)) * 100, 2),
            },
            "assets": [{"name": s, "ret": round(adj_mu[i] * 100, 2), "vol": round(sig_v[i] * 100, 2)}
                       for i, s in enumerate(SVC)],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── AUTO ANALYZER ────────────────────────────────────────────────
@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), user: User = Depends(RoleChecker(["ADMIN"]))):
    """Istalgan datasetni yuklash va avtomatik tahlil"""
    try:
        contents = await file.read()
        result = run_auto_analysis(contents, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── RETRAIN (background) ─────────────────────────────────────────
def _run_pipeline_background(csv_path: str = None):
    """Background threadda pipelineni ishga tushiradi."""
    global _pipeline_status
    _pipeline_status["running"] = True
    _pipeline_status["last_error"] = None
    try:
        from backend.ml_pipeline import IslomiyBankPipeline
        pipe = IslomiyBankPipeline()
        result = pipe.run(csv_path=csv_path)
        # Yangi modelni yuklaymiz
        manager.df = pd.DataFrame()  # runtime df'ni tozalaymiz
        manager.load_from_disk()
        _pipeline_status["last_result"] = result
        print("✅ Background retrain yakunlandi!")
    except Exception as e:
        _pipeline_status["last_error"] = str(e)
        print(f"❌ Background retrain xatosi: {e}")
    finally:
        _pipeline_status["running"] = False


@app.post("/api/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    file: UploadFile = None,
    user: User = Depends(RoleChecker(["ADMIN"]))
):
    """
    Modelni qayta o'qitish (background task).
    - Fayl berilsa → DB ga qo'shib, pipelineni ishga tushiradi
    - Fayl berilmasa → mavjud ma'lumotlar bilan qayta o'qitadi
    """
    global _pipeline_status

    if _pipeline_status["running"]:
        raise HTTPException(status_code=409, detail="Retrain allaqachon ishlayapti. Iltimos kuting.")

    import io, tempfile

    csv_path = None
    stats = {"new": 0, "duplicates": 0}

    if file:
        import pandas as pd
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df_new = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df_new = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Faqat CSV yoki Excel fayllar!")

        stats = manager.ingest_new_data(df_new)

        # Vaqtinchalik fayl (pipeline uchun)
        suffix = ".csv" if file.filename.endswith(".csv") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            csv_path = tmp.name

    background_tasks.add_task(_run_pipeline_background, csv_path)

    return {
        "status":    "started",
        "message":   "Retrain background taskda ishga tushdi",
        "added_rows":      stats["new"],
        "duplicate_rows":  stats["duplicates"],
    }


@app.get("/api/pipeline-status")
async def pipeline_status(user: User = Depends(RoleChecker(["ADMIN"]))):
    """Joriy retrain holati"""
    return {
        "running":     _pipeline_status["running"],
        "last_result": _pipeline_status["last_result"],
        "last_error":  _pipeline_status["last_error"],
        "model_trained": manager._trained,
        "model_name":  manager.meta.get("best_model_name", "—"),
        "auc":         manager.meta.get("best_auc", 0),
        "trained_at":  manager.meta.get("trained_at", "—")[:19] if manager.meta.get("trained_at") else "—",
    }


# ─── IMAGES ───────────────────────────────────────────────────────
@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """saved_models/ papkasidagi PNG fayllarni qaytaradi"""
    allowed = {"eda.png", "monte_carlo.png", "stress_testing.png",
               "model_evaluation.png", "pdp_plots.png", "permutation_importance.png"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Rasm topilmadi")
    path = os.path.join(SAVED_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} hali generatsiya qilinmagan. /api/retrain bajaring.")
    return FileResponse(path, media_type="image/png")


# ─── TRAINING STATS ───────────────────────────────────────────────
@app.get("/api/training-stats")
async def get_training_stats(user: User = Depends(RoleChecker(["ADMIN", "ANALYST", "VIEWER"]))):
    """Model holati va DB statistikasi"""
    db = SessionLocal()
    try:
        count = db.query(CreditApplication).count()
        return {
            "db_row_count":   count,
            "is_trained":     manager._trained,
            "model_name":     manager.meta.get("best_model_name", "—"),
            "ensemble_auc":   manager.meta.get("ensemble_auc", 0),
            "best_auc":       manager.meta.get("best_auc", 0),
            "threshold":      manager.best_thr,
            "trained_at":     manager.meta.get("trained_at", "—")[:19] if manager.meta.get("trained_at") else "—",
            "data_source":    manager.meta.get("data_source", "—"),
            "n_train_samples": manager.meta.get("n_samples", 0),
            "pipeline_running": _pipeline_status["running"],
        }
    finally:
        db.close()


# ─── HISTORY ─────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(limit: int = 50, user: User = Depends(RoleChecker(["ADMIN", "ANALYST"]))):
    """Bazada saqlangan oxirgi bashoratlar"""
    db = SessionLocal()
    try:
        results = (db.query(CreditApplication)
                   .order_by(CreditApplication.timestamp.desc())
                   .limit(limit).all())
        return [
            {
                "id":               r.id,
                "timestamp":        r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "xizmat_turi":      r.xizmat_turi,
                "mintaqa":          r.mintaqa,
                "sektor":           r.sektor,
                "kredit_ball":      r.kredit_ball,
                "moliyalash_miqdori": r.moliyalash_miqdori,
                "pd_qiymati":       r.pd_qiymati,
                "default_holati":   r.default_holati,
                "risk_darajasi":    r.risk_darajasi,
                "sri_indeksi":      r.sri_indeksi,
            }
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
