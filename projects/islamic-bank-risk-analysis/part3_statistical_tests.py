
# ═══════════════════════════════════════════════════════════════
#  QISM 3 — Statistik Testlar (KS, Chi-square, VIF, MI)
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part1_setup_dataset import *
from scipy.stats import jarque_bera  # explicit import so the linter can resolve the name

print("📐 3-Bo'lim: Statistik Testlar boshlandi...")

NUM_FEATS = ['kredit_ball','ltv_nisbati','foyda_stavkasi','bozor_volatilligi',
             'likvidlik','sharia_audit','gharar_darajasi','qarz_xizmat_nisbati',
             'inflyatsiya','valyuta_tebranishi','leverage']

# ── 3.1 Normallik ────────────────────────────────────────────
print('='*65)
print("3.1  NORMALLIK TESTLARI  (Shapiro-Wilk + Jarque-Bera)")
print('-'*65)
col_hdr = "O'zgaruvchi"
print(f"{col_hdr:<26} {'SW p':>9} {'JB p':>9} {'Normal?':>8}")
print('-'*65)
for col in NUM_FEATS:
    smp = df[col].sample(min(300,len(df)),random_state=42)
    _,sw_p = shapiro(smp)
    jb_res = jarque_bera(smp)
    jb_s, jb_p = jb_res[0], jb_res[1]
    nm = 'OK' if sw_p>0.05 and jb_p>0.05 else 'XATO'
    print(f'{col:<26} {sw_p:>9.4f} {jb_p:>9.4f} {nm:>8}')

# ── 3.2 KS testi ─────────────────────────────────────────────
print('\n'+'='*65)
print("3.2  KS TESTI --- Default vs Normal guruh")
print('-'*65)
print(f"{col_hdr:<26} {'KS stat':>9} {'p':>9} {'Farq?':>8}")
print('-'*65)
dg = df[df['default_holati']==1]
ng = df[df['default_holati']==0]
for col in NUM_FEATS:
    ks,p = kstest(dg[col].values,ng[col].values)
    sig = "Ha" if p<0.05 else "Yoq"
    print(f'{col:<26} {ks:>9.4f} {p:>9.4f} {sig:>8}')

# ── 3.3 Chi-square ───────────────────────────────────────────
print('\n'+'='*65)
print('3.3  CHI-SQUARE --- Kategorik x Default')
print('-'*65)
for col in ['xizmat_turi','mintaqa','sektor','garov_sifati','halal_sertifikat']:
    ct = pd.crosstab(df[col],df['default_holati'])
    chi2,p,dof,_ = chi2_contingency(ct)
    sig = 'Muhim' if p<0.05 else 'Muhim emas'
    print(f'{col:<22} chi2={chi2:8.2f}  p={p:.4f}  dof={dof}  {sig}')

# ── 3.4 VIF (manual) ─────────────────────────────────────────
print('\n'+'='*65)
print("3.4  VIF --- Multikolinearlik  (< 5: OK | 5-10: Orta | >10: YUQORI)")
print('-'*65)
X_vif = df[NUM_FEATS].fillna(df[NUM_FEATS].median())
X_vif_sc = (X_vif - X_vif.mean()) / X_vif.std()
vif_results = []
for i,col in enumerate(NUM_FEATS):
    y_vif = X_vif_sc.iloc[:,i].values
    X_other = np.delete(X_vif_sc.values,i,axis=1)
    # R² via lstsq
    X_b = np.column_stack([np.ones(len(X_other)),X_other])
    coef,_,_,_ = np.linalg.lstsq(X_b,y_vif,rcond=None)
    y_hat = X_b @ coef
    ss_res = np.sum((y_vif-y_hat)**2)
    ss_tot = np.sum((y_vif-y_vif.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0
    vif = 1/(1-r2) if r2<0.9999 else 9999
    st = 'OK' if vif<5 else ('ORTA' if vif<10 else 'YUQORI')
    print(f'{col:<26} VIF={vif:8.3f}  {st}')
    vif_results.append((col,vif))

# ── 3.5 Mutual Information ────────────────────────────────────
print('\n'+'='*65)
print("3.5  MUTUAL INFORMATION --- Feature muhimligi (parametrsiz)")
print('-'*65)
mi = mutual_info_classif(
    X_vif.values, df['default_holati'].values, random_state=42
)
mi_df = pd.DataFrame({'Feature':NUM_FEATS,'MI':mi}).sort_values('MI',ascending=False)
for _,row in mi_df.iterrows():
    bar = '#'*int(row['MI']*120)
    feat_name = row["Feature"]
    mi_val = row["MI"]
    print(f'{feat_name:<26} {mi_val:.4f}  {bar}')
print('\n[OK] Statistik testlar yakunlandi.')
print("[OK] Qism 3 muvaffaqiyatli yakunlandi!")
