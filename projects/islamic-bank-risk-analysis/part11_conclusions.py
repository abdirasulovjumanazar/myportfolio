
# ═══════════════════════════════════════════════════════════════
#  QISM 11 — Yakuniy Xulosalar + CSV Eksport
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part9_threshold_eval import *

print("📝 11-Bo'lim: Yakuniy Xulosalar boshlandi...")

best_name = max(ALL_MODELS, key=lambda k: ALL_MODELS[k]['auc'])

print('''
╔═══════════════════════════════════════════════════════════════════════╗
║   ISLOMIY BANK RISK MODELI — YAKUNIY XULOSALAR (v3.0 Standalone)     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  📊 DATASET                                                           ║
║     2 500 tranzaksiya · 31 feature · 4 xizmat turi                   ║
║     AAOIFI, IFSB, Basel III (islomiy) standartlariga muvofiq         ║
║                                                                       ║
║  📐 STATISTIK NATIJALAR                                               ║
║     • KS test: Default/Normal guruh statistik farq qiladi (p<0.05)   ║
║     • Chi-sq: Xizmat turi va Sektor default bilan muhim bog'liq       ║
║     • VIF: Multikolinearlik muammosi yo'q (barcha VIF < 5)           ║
║     • MI: sharia_adj_pd, dsr_ltv, credit_ltv eng muhim feature      ║
║                                                                       ║
║  📉 RISK DARAJASI  (Past → Yuqori)                                    ║
║     Sukuk (SRI-A) < Ijara (SRI-A/B) < Murabaha (B) < Musharaka (C)  ║
║                                                                       ║
║  🎲 MONTE CARLO & STRESS TESTING                                      ║
║     • 20 000 senariy × 252 kun GBM simulatsiyasi                     ║
║     • COVID analog: zarar 5.5x oshadi                                ║
║     • Valyuta inqirozi: volatillik 4x oshadi                         ║
║                                                                       ║
║  🤖 ML MODELLAR                                                        ║
║     • Logistic Regression: GridSearchCV (7 qiymat)                   ║
║     • Random Forest: RandomizedSearchCV (30 iteratsiya)              ║
║     • GBM: RandomizedSearchCV (25 iteratsiya) — Optuna o'rniga       ║
║     • Ensemble (Soft Avg): Eng yuqori AUC                            ║
║     • Manual SMOTE: Class imbalance hal qilindi                      ║
║                                                                       ║
║  🔍 INTERPRETABILITY                                                   ║
║     • Permutation Importance (±std) — 3 model uchun                 ║
║     • PDP (Partial Dependence Plots) — Top-4 feature               ║
║                                                                       ║
║  ✅ TAVSIYALAR                                                         ║
║     1. Sukuk (15→25%) va Ijara (21→30%) ulushini oshiring           ║
║     2. Musharaka uchun kredit ball min: 680+                         ║
║     3. Sharia Audit avtomatik real vaqt monitoring joriy eting       ║
║     4. Optimal classification threshold: ≈ 0.35                      ║
║     5. Og'ir stress senariy kapital zaxirasi: portfelning 12-15%    ║
║     6. LTV chegarasi: Murabaha ≤ 0.70, Ijara ≤ 0.65                ║
╚═══════════════════════════════════════════════════════════════════════╝
''')

# ── CSV eksport ───────────────────────────────────────────────
df.to_csv('islamic_bank_dataset_v3.csv',index=False)
stress_df = pd.read_csv('stress_testing_v3.csv', index_col=0) if os.path.exists('stress_testing_v3.csv') else pd.DataFrame()
if not stress_df.empty:
    stress_df.to_csv('stress_testing_v3.csv')
thr_df.to_csv('threshold_analysis.csv',index=False)

# Model natijalar
model_results = pd.DataFrame([
    {'Model':n, 'AUC':r['auc'],
     'F1':f1_score(y_te,(r['prob']>=BEST_THR).astype(int),zero_division=0),
     'Accuracy':accuracy_score(y_te,(r['prob']>=BEST_THR).astype(int))}
    for n,r in ALL_MODELS.items()
])
model_results.to_csv('model_results.csv',index=False)

print('📁 Saqlangan fayllar:')
saved_files = [
    'islamic_bank_dataset_v3.csv   ← 2500 tranzaksiya, 32 feature',
    'threshold_analysis.csv        ← F1/Prec/Recall vs Threshold',
    'model_results.csv             ← Model metrikalar',
    'eda.png                       ← EDA (8 grafik)',
    'monte_carlo.png               ← GBM simulatsiyasi',
    'stress_testing.png            ← Stress senariylar',
    'permutation_importance.png    ← Interpretability (3 model)',
    'pdp_plots.png                 ← Partial Dependence Plots',
    'model_evaluation.png          ← ROC, CM, Kalibrasiya',
    'portfolio.png                 ← Efficient Frontier',
]
for f in saved_files:
    print(f'  ✅ {f}')

print(f'\n🏆 Eng yaxshi model: {best_name} (AUC={ALL_MODELS[best_name]["auc"]:.4f})')
print('\n🎓 Dissertatsiya loyihasi (v3.0 Standalone) tayyor! 🕌')
print("✅ Qism 11 muvaffaqiyatli yakunlandi!")
