"""
🧪 Bank Integration Test
========================
1C yoki FLEXCUBE rolini o'ynab, integration endpointni sinab ko'rish.

Ishga tushirish:
    python test_integration.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print('═'*55)

def test_health():
    """1. Oddiy health check"""
    print_section("1. API Health Check")
    r = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))

def create_test_bank_and_key():
    """2. Test bank va API kalit yaratish"""
    print_section("2. Test Bank Yaratish")

    # Bank yaratish
    r = requests.post(f"{BASE_URL}/api/v1/integration/admin/banks", params={
        "bank_name": "Test Bank UZ",
        "bank_code": "TEST_UZ",
        "license_type": "TRIAL",
        "contact_email": "test@bank.uz",
        "license_days": 90
    })
    print(f"Bank yaratish: {r.status_code}")
    if r.status_code == 200:
        bank_data = r.json()
        print(json.dumps(bank_data, ensure_ascii=False, indent=2))
        bank_id = bank_data["bank_id"]
    elif r.status_code == 400:
        print("Bank allaqachon mavjud, ID=1 ishlatiladi")
        bank_id = 1
    else:
        print(f"Xato: {r.text}")
        return None

    print_section("3. API Kalit Yaratish")
    r2 = requests.post(f"{BASE_URL}/api/v1/integration/admin/api-keys", params={
        "bank_id": bank_id,
        "description": "1C integration test key",
        "expires_days": 365
    })
    print(f"API Kalit: {r2.status_code}")
    if r2.status_code == 200:
        key_data = r2.json()
        print(json.dumps(key_data, ensure_ascii=False, indent=2))
        return key_data["api_key"]
    else:
        print(f"Xato: {r2.text}")
        return None

def test_integration_score(api_key: str):
    """3. 1C tizimi nomidan kredit arizasi yuborish"""
    print_section("4. Kredit Risk Baholash (1C simulatsiyasi)")

    # 1C yoki FLEXCUBE dan keluvchi ariza formati
    ariza = {
        "product_type": "Murabaha",
        "amount": 75000000,
        "term_months": 36,
        "external_client_id": "1C-CLIENT-00123456",
        "credit_score": 720,
        "age": 38,
        "work_experience": 8,
        "region": "Toshkent",
        "sector": "Savdo",
        "ltv_ratio": 0.60,
        "debt_service_ratio": 0.30,
        "profit_rate": 0.14,
        "liquidity": 0.75,
        "leverage": 0.35,
        "collateral_quality": 4,
        "previous_loans": 2,
        "previous_defaults": 0,
        "sharia_audit_score": 0.90,
        "zakat_paid": 1,
        "halal_certified": 1,
        "gharar_level": 0.08,
        "maysir_exposure": 0.03,
        "partnership_risk": 0,
        "market_volatility": 0.12,
        "gdp_growth": 0.056,
        "inflation": 0.10,
        "currency_fluctuation": 0.05,
        "oil_price_impact": 0.02,
        "bank_sector_index": 0.04
    }

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    r = requests.post(
        f"{BASE_URL}/api/v1/integration/score",
        json=ariza,
        headers=headers
    )

    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        result = r.json()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n✅ Tavsiya: {result['recommendation']}")
        print(f"📊 Risk darajasi: {result['risk_level']}")
        print(f"📈 PD Score: {result['pd_score']:.2%}")
        print(f"🕌 Shariat muvofiq: {'Ha' if result['sharia_compliant'] else 'Yoq'}")
    else:
        print(f"Xato: {r.text}")

def test_wrong_key():
    """4. Noto'g'ri API kalit tekshirish"""
    print_section("5. Noto'g'ri API Kalit Testi")
    headers = {"X-API-Key": "ibra_wrong_key_12345"}
    r = requests.get(f"{BASE_URL}/api/v1/integration/health", headers=headers)
    print(f"Status: {r.status_code} (403 bo'lishi kerak)")
    print(r.json())

def test_no_key():
    """5. API kalit yo'q holat"""
    print_section("6. API Kalit Yo'q Testi")
    r = requests.get(f"{BASE_URL}/api/v1/integration/health")
    print(f"Status: {r.status_code} (401 bo'lishi kerak)")
    print(r.json())

if __name__ == "__main__":
    print("🔌 BANK INTEGRATION TESTI BOSHLANADI...")
    print(f"Server: {BASE_URL}")

    test_health()
    api_key = create_test_bank_and_key()

    if api_key:
        test_integration_score(api_key)

    test_wrong_key()
    test_no_key()

    print("\n✅ BARCHA TESTLAR YAKUNLANDI!")
    print("📖 Swagger UI: http://localhost:8000/docs")
    print("🔌 Integration endpoint: POST /api/v1/integration/score")
