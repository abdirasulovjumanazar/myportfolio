from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timedelta, timezone

def get_tashkent_time():
    """Toshkent vaqtini (UTC+5) qaytaradi"""
    return datetime.now(timezone(timedelta(hours=5)))
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./islamic_bank.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── MULTI-TENANT: Har bir bank alohida ──────────────────────────────────────
class Bank(Base):
    """Har bir bank — alohida tenant"""
    __tablename__ = "banks"

    id            = Column(Integer, primary_key=True, index=True)
    bank_name     = Column(String, unique=True, index=True)   # "Infin Bank"
    bank_code     = Column(String, unique=True, index=True)   # "INFIN_UZ"
    license_type  = Column(String, default="TRIAL")           # TRIAL | STANDARD | ENTERPRISE
    is_active     = Column(Integer, default=1)                # 1=faol, 0=bloklangan
    expires_at    = Column(DateTime, nullable=True)
    contact_email = Column(String, nullable=True)
    created_at    = Column(DateTime, default=get_tashkent_time)


# ─── API KEYS: Bank tizimi (1C, FLEXCUBE) uchun ──────────────────────────────
class ApiKey(Base):
    """Bank tizimi HTTP orqali ulanish uchun API kalitlar"""
    __tablename__ = "api_keys"

    id          = Column(Integer, primary_key=True, index=True)
    bank_id     = Column(Integer, nullable=False, index=True)
    key_hash    = Column(String, unique=True, index=True)   # SHA-256 hash
    key_prefix  = Column(String)                            # Birinchi 8 ta belgi (ko'rsatish uchun)
    description = Column(String, nullable=True)             # "1C integration key"
    is_active   = Column(Integer, default=1)
    last_used   = Column(DateTime, nullable=True)
    created_at  = Column(DateTime, default=get_tashkent_time)
    expires_at  = Column(DateTime, nullable=True)


# ─── KREDIT ARIZALARI ────────────────────────────────────────────────────────
class CreditApplication(Base):
    __tablename__ = "credit_applications"

    id                  = Column(Integer, primary_key=True, index=True)
    bank_id             = Column(Integer, nullable=True, index=True)   # Multi-tenant
    timestamp           = Column(DateTime, default=get_tashkent_time)
    xizmat_turi         = Column(String)
    mintaqa             = Column(String)
    sektor              = Column(String)
    kredit_ball         = Column(Float)
    yosh                = Column(Integer)
    tajriba             = Column(Integer)
    oldingi_kreditlar   = Column(Integer)
    oldingi_defaultlar  = Column(Integer)
    moliyalash_miqdori  = Column(Float)
    muddat_oy           = Column(Integer)
    foyda_stavkasi      = Column(Float)
    ltv_nisbati         = Column(Float)
    qarz_xizmat_nisbati = Column(Float)
    likvidlik           = Column(Float)
    leverage            = Column(Float)
    garov_sifati        = Column(Integer)
    sharia_audit        = Column(Float)
    sharia_score        = Column(Float)
    zakat_status        = Column(Integer)
    partnership_risk    = Column(Integer)
    gharar_darajasi     = Column(Float)
    maysir_ekspozitsiya = Column(Float)
    halal_sertifikat    = Column(Integer)
    bozor_volatilligi   = Column(Float)
    yim_osishi          = Column(Float)
    inflyatsiya         = Column(Float)
    valyuta_tebranishi  = Column(Float)
    neft_narxi          = Column(Float)
    bank_indeksi        = Column(Float)
    pd_qiymati          = Column(Float)
    default_holati      = Column(Integer)
    risk_darajasi       = Column(String)
    sri_indeksi         = Column(Float)
    data_hash           = Column(String, unique=True, index=True)
    # Integration tracking
    source              = Column(String, default="dashboard")  # dashboard | api | 1c | flexcube
    external_client_id  = Column(String, nullable=True)        # Bank tizimidagi mijoz ID


# ─── TRAINING METADATA ───────────────────────────────────────────────────────
class TrainingMetadata(Base):
    __tablename__ = "training_metadata"

    id             = Column(Integer, primary_key=True, index=True)
    version        = Column(String, unique=True)
    timestamp      = Column(DateTime, default=get_tashkent_time)
    row_count      = Column(Integer)
    auc_score      = Column(Float)
    accuracy_score = Column(Float)
    f1_score       = Column(Float)
    details        = Column(Text)


# ─── FOYDALANUVCHILAR ────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    bank_id       = Column(Integer, nullable=True, index=True)   # Multi-tenant
    username      = Column(String, unique=True, index=True)
    password_hash = Column(String)
    role          = Column(String, default="VIEWER")  # ADMIN | ANALYST | VIEWER
    created_at    = Column(DateTime, default=get_tashkent_time)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection verified.")
    except Exception as e:
        print(f"❌ Database Initialization Error: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
