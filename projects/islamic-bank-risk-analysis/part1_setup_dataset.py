
# ═══════════════════════════════════════════════════════════════
#  QISM 1 — Kutubxonalar, Global Sozlamalar va Dataset Generatsiyasi
# ═══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI oynasisiz ishlash uchun
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score,
    precision_score, recall_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline

# Scipy
from scipy import stats
from scipy.stats import (
    kstest, chi2_contingency, shapiro,
    jarque_bera, pearsonr, spearmanr
)
from scipy.optimize import minimize

# Global uslub
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#1B4F72','#A93226','#1E8449','#D4A017','#6C3483']
plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.titleweight': 'bold'
})
np.random.seed(42)

print('✅ Barcha kutubxonalar yuklandi (faqat standart paket)')
print('   numpy | pandas | matplotlib | seaborn | sklearn | scipy')

# ═══════════════════════════════════════════════════════════════
#  Dataset Generatsiyasi (2500 tranzaksiya, 32 feature)
# ═══════════════════════════════════════════════════════════════
N = 2500

SVC_DIST  = {'Murabaha':0.42,'Musharaka':0.22,'Ijara':0.21,'Sukuk':0.15}
SVC_LIST  = list(SVC_DIST.keys())
RISK_P = {
    'Murabaha':  {'pd':0.082,'lgd':0.45,'vol':0.118,'rate':(0.07,0.18),'tenor':[6,12,24,36,60]},
    'Musharaka': {'pd':0.148,'lgd':0.60,'vol':0.245,'rate':(0.12,0.38),'tenor':[12,24,36,60,84,120]},
    'Ijara':     {'pd':0.062,'lgd':0.35,'vol':0.098,'rate':(0.05,0.13),'tenor':[12,24,36,60,84]},
    'Sukuk':     {'pd':0.038,'lgd':0.28,'vol':0.078,'rate':(0.04,0.10),'tenor':[24,36,60,84,120]},
}
REGIONS  = ['Toshkent','Samarqand',"Farg'ona",'Buxoro','Namangan','Qashqadaryo']
SECTORS  = ['Savdo','Ishlab chiqarish',"Qishloq xo'jaligi",'Qurilish','Xizmat','Eksport']

svc_arr = np.random.choice(SVC_LIST, N, p=list(SVC_DIST.values()))
rows = []

for svc in svc_arr:
    p = RISK_P[svc]

    # ─ Mijoz
    cscore   = np.clip(np.random.normal(645,85), 300, 850)
    age      = np.random.randint(22, 65)
    exp_yr   = np.random.randint(1, 25)
    loan_amt = np.random.lognormal(10.8, 1.1)
    tenor    = np.random.choice(p['tenor'])
    rate     = np.random.uniform(*p['rate'])
    region   = np.random.choice(REGIONS)
    sector   = np.random.choice(SECTORS)

    # ─ Risk omillari
    ltv      = np.clip(np.random.beta(4.5,3), 0.10, 0.95)
    dsr      = np.clip(np.random.beta(3,5),   0.05, 0.85)
    col_q    = np.random.choice([1,2,3,4,5], p=[0.05,0.20,0.40,0.25,0.10])
    liq      = np.clip(np.random.beta(7,2.5), 0.20, 1.0)
    lev      = np.random.uniform(0.10, 0.90)
    n_prev   = np.random.poisson(2.5)
    n_def    = np.random.binomial(n_prev, 0.08) if n_prev > 0 else 0

    # ─ Sharia omillari
    sharia   = np.clip(np.random.beta(8,2.5),  0.55, 1.0)
    gharar   = np.clip(np.random.beta(2,8),    0.00, 0.50)
    maysir   = np.clip(np.random.beta(1.5,9),  0.00, 0.40)
    halal    = np.random.choice([0,1], p=[0.12,0.88])

    # ─ Makro
    mkt_vol  = np.abs(np.random.normal(p['vol'], 0.025))
    gdp_g    = np.random.normal(0.056, 0.014)
    inf_r    = np.random.normal(0.098, 0.022)
    fx       = np.abs(np.random.normal(0.048, 0.028))
    oil      = np.random.normal(0.02, 0.15)
    bidx     = np.random.normal(0.04, 0.08)

    # ─ Default ehtimoli (logit)
    z = (-4.2
         + p['pd']*12
         - (cscore-650)*0.006
         + ltv*2.5 + dsr*3.8
         - sharia*2.0 + gharar*4.0 + maysir*3.5
         + inf_r*6.0 + mkt_vol*5.5
         - liq*2.5 + lev*1.8
         + n_def*0.9 - halal*0.6
         - (col_q-3)*0.5
         + np.random.normal(0, 0.25))
    pd_val  = 1/(1+np.exp(-z))
    is_def  = int(np.random.random() < pd_val)

    ead = loan_amt*(1+rate*tenor/12*0.5)
    lgd = np.clip(np.random.normal(p['lgd'],0.08), 0.10, 0.95)
    el  = pd_val*ead*lgd

    if pd_val < 0.10:  rlvl,rcode = 'Past',0
    elif pd_val < 0.25: rlvl,rcode = "O'rta",1
    elif pd_val < 0.45: rlvl,rcode = 'Yuqori',2
    else:               rlvl,rcode = 'Juda Yuqori',3

    rows.append({
        'xizmat_turi':svc,'mintaqa':region,'sektor':sector,
        'kredit_ball':round(cscore),'yosh':age,'tajriba':exp_yr,
        'oldingi_kreditlar':n_prev,'oldingi_defaultlar':n_def,
        'moliyalash_miqdori':round(loan_amt),'muddat_oy':tenor,
        'foyda_stavkasi':round(rate,4),'ltv_nisbati':round(ltv,4),
        'qarz_xizmat_nisbati':round(dsr,4),'likvidlik':round(liq,4),
        'leverage':round(lev,4),'garov_sifati':col_q,
        'sharia_audit':round(sharia,4),'gharar_darajasi':round(gharar,4),
        'maysir_ekspozitsiya':round(maysir,4),'halal_sertifikat':halal,
        'bozor_volatilligi':round(mkt_vol,4),'yim_osishi':round(gdp_g,4),
        'inflyatsiya':round(inf_r,4),'valyuta_tebranishi':round(fx,4),
        'neft_narxi':round(oil,4),'bank_indeksi':round(bidx,4),
        'pd_qiymati':round(pd_val,4),'ead':round(ead),'lgd':round(lgd,4),
        'kutilgan_zarar':round(el),'default_holati':is_def,
        'risk_darajasi':rlvl,'risk_kodi':rcode
    })

df = pd.DataFrame(rows)

# Label encoding
for col,name in [('xizmat_turi','xizmat_enc'),('mintaqa','mintaqa_enc'),('sektor','sektor_enc')]:
    le = LabelEncoder()
    df[name] = le.fit_transform(df[col])

print(f'✅ Dataset: {df.shape[0]} qator × {df.shape[1]} ustun')
print(f'Default nisbati: {df["default_holati"].mean():.2%}')
print()
summary = df.groupby('xizmat_turi')['default_holati'].agg(['count','mean','sum'])
summary.columns = ['N','Default%','Default_N']
summary['Default%'] = summary['Default%'].round(4)
print(summary)
print(df.head(3).to_string())
print("\n✅ Qism 1 muvaffaqiyatli yakunlandi!")
