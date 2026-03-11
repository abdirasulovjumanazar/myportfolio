"""
🔧 Database Migration — bank_id va yangi jadvallar qo'shish
============================================================
Mavjud PostgreSQL bazaga:
  - users jadvaliga bank_id ustuni qo'shadi
  - credit_applications jadvaliga bank_id, source, external_client_id qo'shadi
  - banks yangi jadvali yaratadi
  - api_keys yangi jadvali yaratadi

Ishga tushirish:
    python migrate_bank_integration.py
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./islamic_bank.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)

MIGRATIONS = [
    # ── 1. users jadvaliga bank_id qo'shish ──────────────────────────────
    """
    ALTER TABLE users
    ADD COLUMN IF NOT EXISTS bank_id INTEGER;
    """,

    # ── 2. credit_applications jadvaliga yangi ustunlar ──────────────────
    """
    ALTER TABLE credit_applications
    ADD COLUMN IF NOT EXISTS bank_id INTEGER;
    """,
    """
    ALTER TABLE credit_applications
    ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'dashboard';
    """,
    """
    ALTER TABLE credit_applications
    ADD COLUMN IF NOT EXISTS external_client_id VARCHAR;
    """,

    # ── 3. banks jadvali yaratish ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS banks (
        id            SERIAL PRIMARY KEY,
        bank_name     VARCHAR UNIQUE NOT NULL,
        bank_code     VARCHAR UNIQUE NOT NULL,
        license_type  VARCHAR DEFAULT 'TRIAL',
        is_active     INTEGER DEFAULT 1,
        expires_at    TIMESTAMP,
        contact_email VARCHAR,
        created_at    TIMESTAMP DEFAULT NOW()
    );
    """,

    # ── 4. api_keys jadvali yaratish ──────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id          SERIAL PRIMARY KEY,
        bank_id     INTEGER NOT NULL,
        key_hash    VARCHAR UNIQUE NOT NULL,
        key_prefix  VARCHAR,
        description VARCHAR,
        is_active   INTEGER DEFAULT 1,
        last_used   TIMESTAMP,
        created_at  TIMESTAMP DEFAULT NOW(),
        expires_at  TIMESTAMP
    );
    """,

    # ── 5. Index qo'shish (tezlik uchun) ──────────────────────────────────
    """
    CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_api_keys_bank_id ON api_keys(bank_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_credit_bank_id ON credit_applications(bank_id);
    """,
]

# SQLite uchun alohida SQL (IF NOT EXISTS va SERIAL yo'q)
MIGRATIONS_SQLITE = [
    "ALTER TABLE users ADD COLUMN bank_id INTEGER;",
    "ALTER TABLE credit_applications ADD COLUMN bank_id INTEGER;",
    "ALTER TABLE credit_applications ADD COLUMN source VARCHAR DEFAULT 'dashboard';",
    "ALTER TABLE credit_applications ADD COLUMN external_client_id VARCHAR;",
    """
    CREATE TABLE IF NOT EXISTS banks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bank_name TEXT UNIQUE NOT NULL,
        bank_code TEXT UNIQUE NOT NULL,
        license_type TEXT DEFAULT 'TRIAL',
        is_active INTEGER DEFAULT 1,
        expires_at TIMESTAMP,
        contact_email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bank_id INTEGER NOT NULL,
        key_hash TEXT UNIQUE NOT NULL,
        key_prefix TEXT,
        description TEXT,
        is_active INTEGER DEFAULT 1,
        last_used TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP
    );
    """,
]


def run_migrations():
    is_sqlite = DATABASE_URL.startswith("sqlite")
    migrations = MIGRATIONS_SQLITE if is_sqlite else MIGRATIONS

    print(f"📦 Database: {'SQLite' if is_sqlite else 'PostgreSQL'}")
    print(f"🔗 URL: {DATABASE_URL[:50]}...")
    print()

    with engine.connect() as conn:
        for i, sql in enumerate(migrations, 1):
            sql = sql.strip()
            if not sql:
                continue
            try:
                conn.execute(text(sql))
                conn.commit()
                # Birinchi qismi ko'rsatish
                preview = sql.split('\n')[0].strip()[:60]
                print(f"  ✅ [{i}] {preview}")
            except Exception as e:
                err = str(e).split('\n')[0]
                if "already exists" in err.lower() or "duplicate column" in err.lower():
                    preview = sql.split('\n')[0].strip()[:50]
                    print(f"  ⏭️  [{i}] Allaqachon mavjud: {preview}")
                else:
                    print(f"  ⚠️  [{i}] Xato (o'tkazildi): {err}")

    print()
    print("✅ Migration yakunlandi!")
    print("   Endi serverni ishga tushiring: .\\start.bat")


if __name__ == "__main__":
    run_migrations()
