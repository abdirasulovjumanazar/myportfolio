from pydantic import BaseModel, Field
from typing import Optional, Literal

class CreditInput(BaseModel):
    """Kredit ariza ma'lumotlari"""
    xizmat_turi: Literal['Murabaha', 'Musharaka', 'Ijara', 'Sukuk'] = Field(..., description="Islomiy moliya xizmat turi")
    mintaqa: Literal['Toshkent', 'Samarqand', "Farg'ona", 'Buxoro', 'Namangan', 'Qashqadaryo', 'Andijon', 'Jizzax', 'Navoiy', 'Sirdaryo', 'Surxondaryo', 'Xorazm', "Qoraqalpog'iston"] = 'Toshkent'
    sektor: Literal['Savdo', 'Ishlab chiqarish', "Qishloq xo'jaligi", 'Qurilish', 'Xizmat', 'Eksport'] = 'Savdo'
    kredit_ball: float = Field(650, ge=300, le=850, description="Kredit ball (300-850)")
    yosh: int = Field(35, ge=22, le=65)
    tajriba: int = Field(5, ge=1, le=25)
    oldingi_kreditlar: int = Field(2, ge=0, le=20)
    oldingi_defaultlar: int = Field(0, ge=0, le=10)
    moliyalash_miqdori: float = Field(50000000, ge=1000000, description="UZS")
    muddat_oy: int = Field(24, ge=6, le=120)
    foyda_stavkasi: float = Field(0.12, ge=0.03, le=0.40)
    ltv_nisbati: float = Field(0.65, ge=0.10, le=0.95)
    qarz_xizmat_nisbati: float = Field(0.35, ge=0.05, le=0.85)
    likvidlik: float = Field(0.70, ge=0.20, le=1.0)
    leverage: float = Field(0.40, ge=0.10, le=0.90)
    garov_sifati: int = Field(3, ge=1, le=5)
    sharia_audit: float = Field(0.85, ge=0.55, le=1.0)
    sharia_score: Optional[float] = Field(0.95, ge=0.0, le=1.0)
    zakat_status: int = Field(1, ge=0, le=1)
    partnership_risk: int = Field(1, ge=0, le=2)
    gharar_darajasi: float = Field(0.10, ge=0.0, le=0.50)

    maysir_ekspozitsiya: float = Field(0.05, ge=0.0, le=0.40)
    halal_sertifikat: int = Field(1, ge=0, le=1)
    bozor_volatilligi: float = Field(0.12, ge=0.02, le=0.50)
    yim_osishi: float = Field(0.056, ge=-0.05, le=0.20)
    inflyatsiya: float = Field(0.10, ge=0.01, le=0.30)
    valyuta_tebranishi: float = Field(0.05, ge=0.0, le=0.30)
    neft_narxi: float = Field(0.02, ge=-0.50, le=0.50)
    bank_indeksi: float = Field(0.04, ge=-0.30, le=0.30)

class RiskOutput(BaseModel):
    """Risk tahlili natijasi"""
    pd_qiymati: float
    risk_darajasi: str
    risk_kodi: int
    default_ehtimoli_pct: float
    ead: float
    lgd: float
    kutilgan_zarar: float
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    sri_indeksi: float
    sri_daraja: str
    tavsiya: str
    model_probabilities: dict
    shap_explain: Optional[dict] = None

class StressInput(BaseModel):
    pd_multiplier: float = Field(1.0, ge=1.0, le=10.0)
    vol_multiplier: float = Field(1.0, ge=1.0, le=5.0)
    rate_delta: float = Field(0.0, ge=0.0, le=0.20)

class PortfolioWeights(BaseModel):
    Murabaha: float
    Musharaka: float
    Ijara: float
    Sukuk: float
