
# ═══════════════════════════════════════════════════════════════
#  QISM 7 — GridSearchCV + ML Modellar (LR, RF, GBM, Ensemble)
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part6_feature_smote import *

print("🔧 7-Bo'lim: GridSearchCV Hyperparameter Tuning boshlandi...")
print("⚠️  Bu qism 10-15 daqiqa vaqt olishi mumkin...")

from scipy.stats import randint as sp_randint, uniform as sp_uniform

cv5 = StratifiedKFold(5,shuffle=True,random_state=42)
ALL_MODELS = {}

# ── 1. Logistic Regression ───────────────────────────────────
print('1/3  Logistic Regression ...')
lr_grid = {'C':[0.01,0.05,0.1,0.3,0.5,1.0,2.0]}
lr_gs = GridSearchCV(
    LogisticRegression(max_iter=2000,class_weight='balanced',solver='saga'),
    lr_grid, cv=cv5, scoring='roc_auc', n_jobs=-1
)
lr_gs.fit(X_tr_res,y_tr_res)
lr_best = lr_gs.best_estimator_
lr_prob = lr_best.predict_proba(X_te_sc)[:,1]
lr_auc  = roc_auc_score(y_te,lr_prob)
print(f'   Best C={lr_gs.best_params_["C"]}  CV-AUC={lr_gs.best_score_:.4f}  Test-AUC={lr_auc:.4f}')
ALL_MODELS['Logistic Regression'] = {'model':lr_best,'prob':lr_prob,'auc':lr_auc}

# ── 2. Random Forest (RandomizedSearchCV) ────────────────────
print('2/3  Random Forest (RandomizedSearchCV, n_iter=30) ...')
rf_param_dist = {
    'n_estimators': sp_randint(100,400),
    'max_depth':    sp_randint(4,12),
    'min_samples_leaf': sp_randint(3,15),
    'max_features': ['sqrt','log2',0.5,0.7],
    'min_samples_split': sp_randint(5,20)
}
rf_rs = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced',random_state=42,n_jobs=-1),
    rf_param_dist, n_iter=30, cv=cv5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rf_rs.fit(X_tr_res,y_tr_res)
rf_best = rf_rs.best_estimator_
rf_prob = rf_best.predict_proba(X_te_sc)[:,1]
rf_auc  = roc_auc_score(y_te,rf_prob)
print(f'   Best params: {rf_rs.best_params_}')
print(f'   CV-AUC={rf_rs.best_score_:.4f}  Test-AUC={rf_auc:.4f}')
ALL_MODELS['Random Forest'] = {'model':rf_best,'prob':rf_prob,'auc':rf_auc}

# ── 3. GBM (Optuna o'rniga RandomizedSearchCV) ───────────────
print("3/3  Gradient Boosting (RandomizedSearchCV, n_iter=25) ...")
gbm_param_dist = {
    'n_estimators':   sp_randint(100,400),
    'max_depth':      sp_randint(3,8),
    'learning_rate':  sp_uniform(0.01,0.25),
    'subsample':      sp_uniform(0.5,0.5),
    'min_samples_leaf': sp_randint(5,25),
    'max_features':   ['sqrt','log2',0.6],
}
gbm_rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gbm_param_dist, n_iter=25, cv=cv5, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
gbm_rs.fit(X_tr_res,y_tr_res)
gbm_best = gbm_rs.best_estimator_
gbm_prob = gbm_best.predict_proba(X_te_sc)[:,1]
gbm_auc  = roc_auc_score(y_te,gbm_prob)
print(f'   Best params: {gbm_rs.best_params_}')
print(f'   CV-AUC={gbm_rs.best_score_:.4f}  Test-AUC={gbm_auc:.4f}')
ALL_MODELS['GBM (Tuned)'] = {'model':gbm_best,'prob':gbm_prob,'auc':gbm_auc}

# ── 4. Ensemble ───────────────────────────────────────────────
ens_prob = np.mean([m['prob'] for m in ALL_MODELS.values()],axis=0)
ens_auc  = roc_auc_score(y_te,ens_prob)
ALL_MODELS['Ensemble'] = {'prob':ens_prob,'auc':ens_auc}
print(f'\n4/4  Ensemble (Soft Avg) AUC={ens_auc:.4f}')

# ── Xulosa jadval ─────────────────────────────────────────────
print('\n'+'═'*70)
print(f'{"Model":<22}{"AUC":>8}{"Accuracy":>10}{"F1":>8}{"Prec":>9}{"Recall":>8}')
print('─'*70)
for name,res in ALL_MODELS.items():
    yp = (res['prob']>=0.35).astype(int)
    print(f'{name:<22}{res["auc"]:>8.4f}'
          f'{accuracy_score(y_te,yp):>10.4f}'
          f'{f1_score(y_te,yp,zero_division=0):>8.4f}'
          f'{precision_score(y_te,yp,zero_division=0):>9.4f}'
          f'{recall_score(y_te,yp,zero_division=0):>8.4f}')
print('═'*70)
print("✅ Qism 7 muvaffaqiyatli yakunlandi!")
