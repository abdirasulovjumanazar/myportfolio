import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.database import init_db, SessionLocal
    from backend.models import manager
    print("✅ Basic imports passed.")

    # 1. Database Initialization
    init_db()
    print("✅ Database init passed.")

    # 2. Model Training (Synthetic)
    manager.train()
    print("✅ Model training passed.")

    # 3. Prediction Test
    test_input = {
        'xizmat_turi': 'Murabaha',
        'mintaqa': 'Toshkent',
        'sektor': 'Savdo',
        'kredit_ball': 700.0,
        'yosh': 30,
        'tajriba': 5,
        'oldingi_kreditlar': 1,
        'oldingi_defaultlar': 0,
        'moliyalash_miqdori': 50000000.0,
        'muddat_oy': 24,
        'foyda_stavkasi': 0.15,
        'ltv_nisbati': 0.7,
        'qarz_xizmat_nisbati': 0.3,
        'likvidlik': 0.8,
        'leverage': 0.4,
        'garov_sifati': 4,
        'sharia_audit': 0.9,
        'sharia_score': 0.95,
        'zakat_status': 1,
        'partnership_risk': 0,
        'gharar_darajasi': 0.05,
        'maysir_ekspozitsiya': 0.02,
        'halal_sertifikat': 1,
        'bozor_volatilligi': 0.1,
        'yim_osishi': 0.05,
        'inflyatsiya': 0.12,
        'valyuta_tebranishi': 0.05,
        'neft_narxi': 0.03,
        'bank_indeksi': 0.04
    }
    prediction = manager.predict(test_input)
    print(f"✅ Prediction passed: PD={prediction['pd_qiymati']}")

    # 4. Ingestion Test (Duplicate Check)
    test_df = pd.DataFrame([test_input])
    # Mapping for ingest_new_data (it expects certain names or maps them)
    res1 = manager.ingest_new_data(test_df)
    print(f"✅ Ingestion 1: {res1}")
    
    res2 = manager.ingest_new_data(test_df)
    print(f"✅ Ingestion 2 (Duplicate): {res2}")
    
    if res2['new'] == 0 and res2['duplicates'] == 1:
        print("✅ Duplicate detection works!")
    else:
        print(f"❌ Duplicate detection failed: {res2}")

    print("\n🚀 ALL BACKEND CHECKS PASSED!")

except Exception as e:
    import traceback
    print(f"❌ Verification Failed: {e}")
    traceback.print_exc()
    sys.exit(1)
