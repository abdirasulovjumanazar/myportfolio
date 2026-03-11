
# ═══════════════════════════════════════════════════════════════
#  QISM 6 — Feature Engineering + Manual SMOTE + Train/Test
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part1_setup_dataset import *

print("⚖️ 6-Bo'lim: Feature Engineering + SMOTE boshlandi...")

# ── Feature Engineering ─────────────────────────────────────
df2 = df.copy()
df2['credit_ltv']         = df2['kredit_ball']/(df2['ltv_nisbati']+1e-6)
df2['sharia_adj_pd']      = df2['pd_qiymati']*(1-df2['sharia_audit'])
df2['risk_adj_ret']       = df2['foyda_stavkasi']/(df2['bozor_volatilligi']+1e-6)
df2['gharar_maysir']      = df2['gharar_darajasi']+df2['maysir_ekspozitsiya']
df2['el_per_unit']        = df2['kutilgan_zarar']/(df2['moliyalash_miqdori']+1e-6)
df2['dsr_ltv']            = df2['qarz_xizmat_nisbati']*df2['ltv_nisbati']
df2['macro_stress']       = df2['inflyatsiya']+df2['valyuta_tebranishi']-df2['yim_osishi']
df2['prev_default_rate']  = df2['oldingi_defaultlar']/(df2['oldingi_kreditlar']+1e-6)

FEATURES = [
    'xizmat_enc','mintaqa_enc','sektor_enc',
    'kredit_ball','ltv_nisbati','foyda_stavkasi','muddat_oy',
    'qarz_xizmat_nisbati','likvidlik','leverage','garov_sifati',
    'sharia_audit','gharar_darajasi','maysir_ekspozitsiya','halal_sertifikat',
    'bozor_volatilligi','yim_osishi','inflyatsiya','valyuta_tebranishi',
    'neft_narxi','bank_indeksi','oldingi_kreditlar','oldingi_defaultlar',
    'credit_ltv','sharia_adj_pd','risk_adj_ret','gharar_maysir',
    'el_per_unit','dsr_ltv','macro_stress','prev_default_rate'
]
TARGET = 'default_holati'

X = df2[FEATURES].fillna(df2[FEATURES].median())
y = df2[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

# ── Manual SMOTE (sklearn yo'q, scipy bor) ──────────────────
def manual_smote(X,y,k=5,ratio=1.0,seed=42):
    """Sodda SMOTE implementatsiyasi — faqat numpy bilan"""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(seed)
    min_cls = int(y.sum() < (y==0).sum())
    X_min = X[y==min_cls]; X_maj = X[y!=min_cls]
    n_need = int(len(X_maj)*ratio) - len(X_min)
    if n_need <= 0:
        return X, y
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_min)
    _,inds = nn.kneighbors(X_min)
    synthetic = []
    for _ in range(n_need):
        i = rng.randint(0,len(X_min))
        nb = inds[i,rng.randint(1,k+1)]
        lam = rng.random()
        synthetic.append(X_min[i] + lam*(X_min[nb]-X_min[i]))
    X_new = np.vstack([X,np.array(synthetic)])
    y_new = np.concatenate([y, np.full(n_need,min_cls)])
    return X_new, y_new

X_tr_res, y_tr_res = manual_smote(X_tr_sc, y_tr.values)
print(f'Train (original): {dict(zip(*np.unique(y_tr,return_counts=True)))}')
print(f'Train (SMOTE):    {dict(zip(*np.unique(y_tr_res,return_counts=True)))}')
print(f'\nFeatures: {len(FEATURES)} | Train: {len(X_tr_res)} | Test: {len(X_te)}')
print("✅ Qism 6 muvaffaqiyatli yakunlandi!")
