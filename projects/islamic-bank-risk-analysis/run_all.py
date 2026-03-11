
#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  🕌 ISLOMIY BANK RISK TAHLILI — BARCHA QISMLARNI ISHGA TUSHIRUVCHI
#  Fayllar: part1 → part11 (ketma-ket)
# ═══════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os
import time

# Script fayllari joylashgan papka
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Ishchi papkani o'zgartirish (PNG fayllar shu yerga saqlanadi)

PARTS = [
    ("QISM 1: Kutubxonalar + Dataset Generatsiyasi",          "part1_setup_dataset.py"),
    ("QISM 2: EDA Vizualizatsiya",                             "part2_eda.py"),
    ("QISM 3: Statistik Testlar (KS, Chi-sq, VIF, MI)",       "part3_statistical_tests.py"),
    ("QISM 4: Risk Ko'rsatkichlari (VaR/CVaR/SRI/Basel)",     "part4_risk_metrics.py"),
    ("QISM 5: Monte Carlo GBM + Stress Testing",              "part5_monte_carlo.py"),
    ("QISM 6: Feature Engineering + SMOTE",                   "part6_feature_smote.py"),
    ("QISM 7: ML Modellar (LR, RF, GBM, Ensemble)",           "part7_ml_models.py"),
    ("QISM 8: Model Interpretability (Perm. Imp + PDP)",      "part8_interpretability.py"),
    ("QISM 9: Threshold Opt + Model Baholash Dashboard",      "part9_threshold_eval.py"),
    ("QISM 10: Bootstrap CI + Portfel Optimizatsiyasi",       "part10_bootstrap_portfolio.py"),
    ("QISM 11: Yakuniy Xulosalar + CSV Eksport",              "part11_conclusions.py"),
]

def run_part(name, script):
    """Bitta qismni ishga tushirish va natijani chiqarish"""
    print(f"\n{'═'*70}")
    print(f"▶  {name}")
    print(f"   Fayl: {script}")
    print(f"{'─'*70}")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    elapsed = time.time() - start
    
    if result.stdout:
        print(result.stdout)
    if result.returncode == 0:
        print(f"✅  {name} — {elapsed:.1f} soniyada yakunlandi")
        return True
    else:
        print(f"❌  XATO: {name}")
        if result.stderr:
            print("XATO MATNI:")
            print(result.stderr[-2000:])  # Oxirgi 2000 belgi
        return False

def main():
    print("🕌 ISLOMIY BANK RISK TAHLILI — v3.0 Standalone")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Papka:  {SCRIPT_DIR}")
    print(f"   Jami qismlar: {len(PARTS)}")
    
    total_start = time.time()
    results = []
    
    for name, script in PARTS:
        success = run_part(name, script)
        results.append((name, success))
        if not success:
            print(f"\n⚠️  DIQQAT: '{name}' muvaffaqiyatsiz tugadi!")
            print("   Keyingi qismlar baribir ishga tushiriladi...")
    
    # Yakuniy hisobot
    total_time = time.time() - total_start
    success_count = sum(1 for _, s in results if s)
    
    print(f"\n{'═'*70}")
    print(f"📋 YAKUNIY HISOBOT — {total_time:.1f} soniya")
    print(f"{'─'*70}")
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status}  {name}")
    
    print(f"\n{'─'*70}")
    print(f"  Muvaffaqiyatli: {success_count}/{len(PARTS)} qism")
    
    if success_count == len(PARTS):
        print("\n🎉 BARCHA QISMLAR MUVAFFAQIYATLI YAKUNLANDI! 🕌")
    else:
        failed = [n for n, s in results if not s]
        print(f"\n⚠️  Muvaffaqiyatsiz qismlar: {', '.join(failed)}")
    
    print('═'*70)

if __name__ == "__main__":
    main()
