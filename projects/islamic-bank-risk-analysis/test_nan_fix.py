import pandas as pd
import numpy as np
from backend.models import manager

def test_nan_fix():
    print("🚀 Testing NaN fix in Online Learning...")
    
    # 1. Initialize manager (generates initial dataset)
    manager.train()
    initial_size = len(manager.df)
    print(f"Initial dataset size: {initial_size}")
    
    # 2. Create a "bad" dataset with NaNs in the target column
    # And also use a different name for the target column to test smart mapping
    bad_data = pd.DataFrame({
        'target': [1, 0, np.nan, 1, np.nan], # 2 NaNs here
        'kredit_ball': [700, 600, 500, 800, 400],
        'xizmat_turi': ['Murabaha', 'Murabaha', 'Ijara', 'Ijara', 'Sukuk'],
        'mintaqa': ['Toshkent', 'Samarqand', 'Samarqand', 'Toshkent', 'Buxoro'],
        'sektor': ['Savdo', 'Savdo', 'Xizmat', 'Xizmat', 'Savdo'],
        'moliyalash_miqdori': [1000000, 2000000, 1500000, 3000000, 500000],
        'muddat_oy': [12, 24, 12, 36, 12],
        'foyda_stavkasi': [0.15, 0.12, 0.10, 0.18, 0.08],
        'ltv_nisbati': [0.7, 0.8, 0.6, 0.9, 0.5],
        'qarz_xizmat_nisbati': [0.3, 0.4, 0.2, 0.5, 0.1],
        'likvidlik': [0.8, 0.7, 0.9, 0.6, 1.0],
        'leverage': [0.5, 0.6, 0.4, 0.7, 0.3],
        'garov_sifati': [3, 4, 3, 5, 2],
        'sharia_audit': [0.9, 0.8, 1.0, 0.7, 0.95],
        'sharia_score': [0.9, 0.85, 0.95, 0.75, 0.98],
        'gharar_darajasi': [0.05, 0.1, 0.02, 0.15, 0.01],
        'maysir_ekspozitsiya': [0.02, 0.05, 0.01, 0.1, 0.01],
        'halal_sertifikat': [1, 1, 1, 1, 1],
        'zakat_status': [1, 1, 1, 0, 1],
        'partnership_risk': [0, 1, 0, 2, 0],
        'bozor_volatilligi': [0.1, 0.15, 0.08, 0.2, 0.05],
        'yim_osishi': [0.05, 0.04, 0.06, 0.03, 0.07],
        'inflyatsiya': [0.1, 0.12, 0.09, 0.15, 0.08],
        'valyuta_tebranishi': [0.05, 0.07, 0.04, 0.1, 0.02],
        'neft_narxi': [0.02, -0.05, 0.01, -0.1, 0.03],
        'bank_indeksi': [0.05, 0.02, 0.08, -0.03, 0.1],
        'oldingi_kreditlar': [3, 10, 1, 5, 0],
        'oldingi_defaultlar': [0, 1, 0, 1, 0]
    })
    
    try:
        # 3. Try to ingest the bad data
        manager.ingest_new_data(bad_data)
        print("✅ Fix works! No 'Input y contains NaN' error.")
        print(f"Final dataset size: {len(manager.df)}")
    except Exception as e:
        print(f"❌ Fix FAILED: {e}")

if __name__ == "__main__":
    test_nan_fix()
