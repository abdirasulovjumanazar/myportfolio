"""
🔐 Authentication Module — Production-Ready
JWT + HttpOnly Cookie + Role-Based Access Control
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .database import get_db, User

# ─── Security Configuration ────────────────────────────────────────
# SECRET_KEY must be set in .env for production. No hard-coded fallback.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    import secrets
    _generated = secrets.token_hex(32)
    print(f"⚠️  WARNING: SECRET_KEY not set in .env. "
          f"Using auto-generated key (sessions won't survive restarts)...")
    SECRET_KEY = _generated

ALGORITHM = "HS256"
# Token lifetime — configurable via .env (default 24 hours for usability)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("TOKEN_EXPIRE_MINUTES", 60 * 24))


# ─── Password Utilities ─────────────────────────────────────────────
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ─── JWT Token Utilities ────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ─── User Authentication ─────────────────────────────────────────────
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    return user


# ─── Dependency: Current User ────────────────────────────────────────
async def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """Extract and validate JWT from HttpOnly cookie."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Seans topilmadi. Iltimos, tizimga qayta kiring.",
        )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Noto'g'ri token tarkibi.")
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Seans muddati tugagan yoki token noto'g'ri. Qayta kiring."
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Foydalanuvchi topilmadi.")
    return user


# ─── Role-Based Access Control ───────────────────────────────────────
class RoleChecker:
    """Dependency for endpoint-level role enforcement."""

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Bu amalni bajarish uchun huquqingiz yetarli emas. "
                    f"Kerakli rol: {', '.join(self.allowed_roles)}"
                ),
            )
        return user
