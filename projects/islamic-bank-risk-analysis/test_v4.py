import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint):
    print(f"Testing {endpoint}...")
    try:
        r = requests.get(f"{BASE_URL}{endpoint}")
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 30)

def test_predict():
    print("Testing /api/predict...")
    payload = {
        "xizmat_turi": "Murabaha",
        "mintaqa": "Toshkent",
        "sektor": "Savdo",
        "kredit_ball": 650,
        "yosh": 35,
        "tajriba": 5,
        "oldingi_kreditlar": 2,
        "oldingi_defaultlar": 0,
        "moliyalash_miqdori": 50000000,
        "muddat_oy": 24,
        "foyda_stavkasi": 0.12,
        "ltv_nisbati": 0.65,
        "qarz_xizmat_nisbati": 0.35,
        "likvidlik": 0.70,
        "leverage": 0.40,
        "garov_sifati": 3,
        "sharia_audit": 0.85,
        "sharia_score": 0.85,
        "zakat_status": 1,
        "partnership_risk": 0,
        "gharar_darajasi": 0.10,
        "maysir_ekspozitsiya": 0.05,
        "halal_sertifikat": 1,
        "bozor_volatilligi": 0.12,
        "yim_osishi": 0.056,
        "inflyatsiya": 0.10,
        "valyuta_tebranishi": 0.05,
        "neft_narxi": 0.02,
        "bank_indeksi": 0.04
    }
    try:
        r = requests.post(f"{BASE_URL}/api/predict", json=payload)
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(r.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 30)

if __name__ == "__main__":
    test_endpoint("/api/health")
    test_endpoint("/api/training-stats")
    test_predict()
    test_endpoint("/api/pipeline-status")
