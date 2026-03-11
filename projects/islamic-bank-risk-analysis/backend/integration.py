"""
🔌 Bank Integration Adapter
=============================
Tashqi bank tizimlari (1C:Bank, Oracle FLEXCUBE, IBSO) dan
HTTP orqali kredit arizalarini qabul qilish va risk baholash.

Ishlash tartibi:
  [1C / FLEXCUBE]  --POST /api/v1/integration/score--> [Bu modul] --> [ML Engine] --> JSON javob
"""
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db, ApiKey, Bank, CreditApplication, get_tashkent_time

router = APIRouter(prefix="/api/v1/integration", tags=["🔌 Bank Integration"])


# ─── SCHEMAS ─────────────────────────────────────────────────────────────────

class BankScoreRequest(BaseModel):
    """
    Bank tizimidan keluvchi kredit ariza formati.
    1C, FLEXCUBE yoki boshqa tizimlar shu formatda yuboradi.
    """
    # Majburiy maydonlar
    product_type: str = Field(..., description="Islom xizmat turi: Murabaha | Musharaka | Ijara | Sukuk")
    amount: float = Field(..., description="Moliyalash miqdori (UZS)")
    term_months: int = Field(..., description="Muddat (oylar)")

    # Mijoz ma'lumotlari
    external_client_id: Optional[str] = Field(None, description="Bank tizimidagi mijoz ID (1C kod yoki FLEXCUBE ID)")
    credit_score: Optional[float] = Field(650, description="Kredit bali (300-850)")
    age: Optional[int] = Field(35, description="Mijoz yoshi")
    work_experience: Optional[int] = Field(5, description="Ish tajribasi (yil)")
    region: Optional[str] = Field("Toshkent", description="Mintaqa")
    sector: Optional[str] = Field("Savdo", description="Iqtisodiy sektor")

    # Moliyaviy ko'rsatkichlar
    ltv_ratio: Optional[float] = Field(0.65, description="Loan-to-Value nisbati")
    debt_service_ratio: Optional[float] = Field(0.35, description="Qarz xizmat nisbati")
    profit_rate: Optional[float] = Field(0.12, description="Foyda stavkasi")
    liquidity: Optional[float] = Field(0.70)
    leverage: Optional[float] = Field(0.40)
    collateral_quality: Optional[int] = Field(3, description="Garov sifati (1-5)")
    previous_loans: Optional[int] = Field(0)
    previous_defaults: Optional[int] = Field(0)

    # Shariat ko'rsatkichlari
    sharia_audit_score: Optional[float] = Field(0.85)
    zakat_paid: Optional[int] = Field(1, description="Zakat to'langan: 1=ha, 0=yo'q")
    halal_certified: Optional[int] = Field(1)
    gharar_level: Optional[float] = Field(0.10)
    maysir_exposure: Optional[float] = Field(0.05)
    partnership_risk: Optional[int] = Field(0, description="0=past, 1=o'rta, 2=yuqori")

    # Makroiqtisodiy
    market_volatility: Optional[float] = Field(0.12)
    gdp_growth: Optional[float] = Field(0.056)
    inflation: Optional[float] = Field(0.10)
    currency_fluctuation: Optional[float] = Field(0.05)
    oil_price_impact: Optional[float] = Field(0.02)
    bank_sector_index: Optional[float] = Field(0.04)


class BankScoreResponse(BaseModel):
    """Risk baholash natijasi — bank tizimiga qaytariladigan javob"""
    # Asosiy natija
    recommendation: str           # APPROVE | REVIEW | REJECT
    risk_level: str               # LOW | MEDIUM | HIGH | CRITICAL
    pd_score: float               # Default ehtimolligi (0.0 – 1.0)
    sri_index: float              # Shariat Risk Indeksi (0.0 – 1.0, yuqori=yaxshi)

    # Tafsilotlar
    sharia_compliant: bool        # Shariat muvofiq?
    confidence: float             # Model ishonch darajasi (0.0 – 1.0)
    expected_loss_uzs: float      # Kutilgan zarar (UZS)

    # Meta
    external_client_id: Optional[str]
    processed_at: str             # ISO timestamp
    model_version: str            # ML model versiyasi
    source_system: str            # Qaysi tizimdan keldi

    # Tavsiyalar
    notes: list[str]              # Qo'shimcha izohlar


# ─── API KEY HELPER ───────────────────────────────────────────────────────────

def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """
    Yangi API kalit yaratadi.
    Returns: (raw_key, key_hash)
    raw_key faqat bir marta ko'rsatiladi, keyin saqlanmaydi!
    """
    raw_key = "ibra_" + secrets.token_urlsafe(32)  # ibra = Islamic Bank Risk API
    key_hash = hash_api_key(raw_key)
    return raw_key, key_hash


async def get_api_key_bank(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db)
) -> Bank:
    """
    X-API-Key header orqali bank autentifikatsiyasi.
    Bank tizimi har bir so'rovga shu headerni qo'shadi.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header talab qilinadi",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    key_hash = hash_api_key(x_api_key)
    api_key = db.query(ApiKey).filter(
        ApiKey.key_hash == key_hash,
        ApiKey.is_active == 1
    ).first()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Yaroqsiz yoki nofaol API kalit"
        )

    # Muddati o'tganmi?
    if api_key.expires_at and api_key.expires_at < get_tashkent_time():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API kalit muddati tugagan. Yangi kalit oling."
        )

    # Oxirgi ishlatilgan vaqtni yangilash
    api_key.last_used = get_tashkent_time()
    db.commit()

    # Bank ma'lumotlarini oling
    bank = db.query(Bank).filter(
        Bank.id == api_key.bank_id,
        Bank.is_active == 1
    ).first()

    if not bank:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bank litsenziyasi faol emas"
        )

    # Bank litsenziyasi muddati
    if bank.expires_at and bank.expires_at < get_tashkent_time():
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Bank litsenziyasi muddati tugagan: {bank.expires_at.date()}"
        )

    return bank


# ─── INTEGRATION ENDPOINTS ───────────────────────────────────────────────────

@router.post("/score", response_model=BankScoreResponse, summary="Kredit risk baholash (Bank API)")
async def score_credit_risk(
    request: BankScoreRequest,
    bank: Bank = Depends(get_api_key_bank),
    db: Session = Depends(get_db)
):
    """
    ## Bank tizimidan kredit arizasini qabul qilib, risk baholaydi.

    **Autentifikatsiya:** `X-API-Key: ibra_xxxxxxxx` header yuborilishi kerak.

    **Ishlash tartibi:**
    1. Bank tizimi (1C, FLEXCUBE) HTTP POST yuboradi
    2. API kalit tekshiriladi
    3. ML modeli risk baholaydi
    4. JSON javob qaytariladi

    **Recommendation qiymatlari:**
    - `APPROVE` — tasdiqlash tavsiya qilinadi (PD < 20%)
    - `REVIEW` — qo'shimcha tekshiruv kerak (PD 20–50%)
    - `REJECT` — rad etish tavsiya qilinadi (PD > 50%)
    """
    from backend.models import manager

    # ML modeli yuklanganmi?
    if not manager._trained:
        ok = manager.load_from_disk()
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML modeli hali o'qitilmagan. /api/retrain endpoint orqali o'qiting."
            )

    # Bank formatini ML formatiga o'girish
    ml_input = {
        "xizmat_turi":          request.product_type,
        "mintaqa":              request.region,
        "sektor":               request.sector,
        "kredit_ball":          request.credit_score,
        "yosh":                 request.age,
        "tajriba":              request.work_experience,
        "oldingi_kreditlar":    request.previous_loans,
        "oldingi_defaultlar":   request.previous_defaults,
        "moliyalash_miqdori":   request.amount,
        "muddat_oy":            request.term_months,
        "foyda_stavkasi":       request.profit_rate,
        "ltv_nisbati":          request.ltv_ratio,
        "qarz_xizmat_nisbati":  request.debt_service_ratio,
        "likvidlik":            request.liquidity,
        "leverage":             request.leverage,
        "garov_sifati":         request.collateral_quality,
        "sharia_audit":         request.sharia_audit_score,
        "zakat_status":         request.zakat_paid,
        "partnership_risk":     request.partnership_risk,
        "gharar_darajasi":      request.gharar_level,
        "maysir_ekspozitsiya":  request.maysir_exposure,
        "halal_sertifikat":     request.halal_certified,
        "bozor_volatilligi":    request.market_volatility,
        "yim_osishi":           request.gdp_growth,
        "inflyatsiya":          request.inflation,
        "valyuta_tebranishi":   request.currency_fluctuation,
        "neft_narxi":           request.oil_price_impact,
        "bank_indeksi":         request.bank_sector_index,
        "halal_status":         request.halal_certified,
    }

    try:
        result = manager.predict(ml_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML model xatosi: {str(e)}")

    pd_score    = result.get("pd_qiymati", 0.5)
    risk_level  = result.get("risk_darajasi", "MEDIUM")
    sri_index   = result.get("sri_indeksi", 0.5)

    # Tavsiya aniqlash
    if pd_score < 0.20:
        recommendation = "APPROVE"
    elif pd_score < 0.50:
        recommendation = "REVIEW"
    else:
        recommendation = "REJECT"

    # Shariat muvofiqlik
    sharia_ok = (
        (request.sharia_audit_score or 0.85) >= 0.70 and
        (request.gharar_level or 0) <= 0.25 and
        (request.maysir_exposure or 0) <= 0.20 and
        (request.halal_certified or 1) == 1
    )

    # Izohlar
    notes = []
    if pd_score > 0.30:
        notes.append("⚠️ Default ehtimolligi yuqori — qo'shimcha garov talab qiling")
    if (request.gharar_level or 0) > 0.20:
        notes.append("⚠️ Gharar darajasi yuqori — shariat auditorini jalb qiling")
    if (request.previous_defaults or 0) > 0:
        notes.append(f"⚠️ Mijozda {request.previous_defaults} ta oldingi default mavjud")
    if sri_index < 0.60:
        notes.append("⚠️ Shariat Risk Indeksi past — muvofiqligi to'liq tekshirilmagan")
    if not notes:
        notes.append("✅ Barcha risk ko'rsatkichlari normal chegarada")

    # Kutilgan zarar (EL = PD × LGD × EAD)
    lgd = 0.45
    el_uzs = pd_score * lgd * request.amount

    # Bazaga saqlash (bank_id bilan)
    import hashlib, json as _json
    data_str = _json.dumps({**ml_input, "bank_id": bank.id}, sort_keys=True)
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()

    existing = db.query(CreditApplication).filter(
        CreditApplication.data_hash == data_hash
    ).first()

    if not existing:
        app = CreditApplication(
            bank_id=bank.id,
            xizmat_turi=request.product_type,
            mintaqa=request.region or "Toshkent",
            sektor=request.sector or "Savdo",
            kredit_ball=request.credit_score,
            yosh=request.age,
            tajriba=request.work_experience,
            oldingi_kreditlar=request.previous_loans,
            oldingi_defaultlar=request.previous_defaults,
            moliyalash_miqdori=request.amount,
            muddat_oy=request.term_months,
            foyda_stavkasi=request.profit_rate,
            ltv_nisbati=request.ltv_ratio,
            qarz_xizmat_nisbati=request.debt_service_ratio,
            likvidlik=request.liquidity,
            leverage=request.leverage,
            garov_sifati=request.collateral_quality,
            sharia_audit=request.sharia_audit_score,
            zakat_status=request.zakat_paid,
            partnership_risk=request.partnership_risk,
            gharar_darajasi=request.gharar_level,
            maysir_ekspozitsiya=request.maysir_exposure,
            halal_sertifikat=request.halal_certified,
            bozor_volatilligi=request.market_volatility,
            yim_osishi=request.gdp_growth,
            inflyatsiya=request.inflation,
            valyuta_tebranishi=request.currency_fluctuation,
            neft_narxi=request.oil_price_impact,
            bank_indeksi=request.bank_sector_index,
            pd_qiymati=pd_score,
            risk_darajasi=risk_level,
            sri_indeksi=sri_index,
            data_hash=data_hash,
            source="api",
            external_client_id=request.external_client_id,
        )
        db.add(app)
        db.commit()

    return BankScoreResponse(
        recommendation=recommendation,
        risk_level=risk_level,
        pd_score=round(pd_score, 4),
        sri_index=round(sri_index, 4),
        sharia_compliant=sharia_ok,
        confidence=round(result.get("confidence", 0.85), 4),
        expected_loss_uzs=round(el_uzs, 2),
        external_client_id=request.external_client_id,
        processed_at=get_tashkent_time().isoformat(),
        model_version=f"v{get_tashkent_time().strftime('%Y%m')}",
        source_system=f"{bank.bank_code} via API",
        notes=notes,
    )


@router.get("/health", summary="Integration servis holati")
async def integration_health(bank: Bank = Depends(get_api_key_bank)):
    """Bank tizimi ulanishni tekshirish uchun."""
    return {
        "status": "ok",
        "bank": bank.bank_name,
        "bank_code": bank.bank_code,
        "license": bank.license_type,
        "timestamp": get_tashkent_time().isoformat(),
    }


# ─── ADMIN ENDPOINTS (API Key boshqaruvi) ────────────────────────────────────

@router.post("/admin/banks", summary="Yangi bank qo'shish (ADMIN)")
async def create_bank(
    bank_name: str,
    bank_code: str,
    license_type: str = "TRIAL",
    contact_email: Optional[str] = None,
    license_days: int = 30,
    db: Session = Depends(get_db)
):
    """Yangi bank tenant yaratish (faqat super-admin ishlatadi)."""
    existing = db.query(Bank).filter(Bank.bank_code == bank_code).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Bank kodi allaqachon mavjud: {bank_code}")

    bank = Bank(
        bank_name=bank_name,
        bank_code=bank_code.upper(),
        license_type=license_type,
        contact_email=contact_email,
        expires_at=get_tashkent_time() + timedelta(days=license_days),
    )
    db.add(bank)
    db.commit()
    db.refresh(bank)
    return {"bank_id": bank.id, "bank_code": bank.bank_code, "expires_at": bank.expires_at}


@router.post("/admin/api-keys", summary="Bank uchun API kalit yaratish (ADMIN)")
async def create_api_key(
    bank_id: int,
    description: str = "Integration key",
    expires_days: int = 365,
    db: Session = Depends(get_db)
):
    """
    Bank uchun yangi API kalit yaratish.
    **Muhim:** Raw kalit faqat bir marta ko'rsatiladi — saqlab qo'ying!
    """
    bank = db.query(Bank).filter(Bank.id == bank_id).first()
    if not bank:
        raise HTTPException(status_code=404, detail="Bank topilmadi")

    raw_key, key_hash = generate_api_key()

    api_key = ApiKey(
        bank_id=bank_id,
        key_hash=key_hash,
        key_prefix=str(raw_key)[:12] + "...",
        description=description,
        expires_at=get_tashkent_time() + timedelta(days=expires_days),
    )
    db.add(api_key)
    db.commit()

    return {
        "message": "✅ API kalit yaratildi. Bu kalitni xavfsiz joyda saqlang!",
        "api_key": raw_key,           # Faqat bir marta ko'rsatiladi!
        "key_prefix": api_key.key_prefix,
        "bank": bank.bank_name,
        "expires_at": api_key.expires_at,
        "usage": {
            "header": "X-API-Key",
            "example": f"curl -H 'X-API-Key: {raw_key}' http://your-server/api/v1/integration/score"
        }
    }


@router.get("/admin/banks", summary="Barcha banklar ro'yxati (ADMIN)")
async def list_banks(db: Session = Depends(get_db)):
    banks = db.query(Bank).all()
    return [
        {
            "id": b.id,
            "bank_name": b.bank_name,
            "bank_code": b.bank_code,
            "license_type": b.license_type,
            "is_active": b.is_active,
            "expires_at": b.expires_at,
        }
        for b in banks
    ]
