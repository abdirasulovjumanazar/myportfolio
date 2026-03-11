
# ═══════════════════════════════════════════════════════════════
#  QISM 8 — Permutation Importance + PDP  (SHAP o'rniga)
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part7_ml_models import *

print("🧠 8-Bo'lim: Model Interpretability boshlandi...")

# ── 8.1 Permutation Importance (3 model) ────────────────────
fig,axes = plt.subplots(1,3,figsize=(20,7))

perm_dfs = {}
for ax,(name,res) in zip(axes,list(ALL_MODELS.items())[:3]):
    if 'model' not in res: continue
    pi = permutation_importance(
        res['model'], X_te_sc, y_te,
        n_repeats=15, random_state=42,
        scoring='roc_auc', n_jobs=-1
    )
    pdf = pd.DataFrame({'feature':FEATURES,'imp':pi.importances_mean,
                         'std':pi.importances_std}).sort_values('imp',ascending=True).tail(15)
    perm_dfs[name] = pdf

    ax.barh(range(len(pdf)), pdf['imp'], xerr=pdf['std'],
            color=PALETTE[list(ALL_MODELS.keys()).index(name)],
            alpha=0.85, edgecolor='white',
            error_kw={'elinewidth':1.2,'capsize':3})
    ax.set_yticks(range(len(pdf)))
    ax.set_yticklabels([f[:18] for f in pdf['feature']],fontsize=8)
    ax.set_title(f'Permutation Importance\n{name}',fontweight='bold')
    ax.set_xlabel('AUC pasayishi')
    ax.axvline(0,color='black',lw=0.8,ls='--')

plt.tight_layout()
plt.savefig('permutation_importance.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ permutation_importance.png saqlandi.')

# ── 8.2 Partial Dependence Plot (PDP) ───────────────────────
# Top-4 feature (RF Permutation asosida)
if 'Random Forest' in perm_dfs:
    top4 = perm_dfs['Random Forest'].sort_values('imp',ascending=False).head(4)['feature'].tolist()
else:
    top4 = ['sharia_adj_pd','credit_ltv','gharar_maysir','qarz_xizmat_nisbati']

fig,axes = plt.subplots(2,2,figsize=(14,10))
best_mdl = ALL_MODELS.get('GBM (Tuned)',ALL_MODELS.get('Random Forest'))

for ax,feat in zip(axes.flatten(),top4):
    feat_idx = FEATURES.index(feat)
    feat_vals = X_te_sc[:,feat_idx]
    grid = np.linspace(feat_vals.min(),feat_vals.max(),50)

    pdp_vals = []
    for gv in grid:
        X_mod = X_te_sc.copy()
        X_mod[:,feat_idx] = gv
        pdp_vals.append(best_mdl['model'].predict_proba(X_mod)[:,1].mean())

    pdp_arr = np.array(pdp_vals)

    # Asl scale qaytarish
    feat_orig = df2[feat].fillna(df2[feat].median())
    fmean = feat_orig.mean(); fstd = feat_orig.std()
    grid_orig = grid*fstd + fmean

    ax.plot(grid_orig,pdp_arr*100,color=PALETTE[1],lw=2.5)
    ax.fill_between(grid_orig,pdp_arr*100,alpha=0.15,color=PALETTE[1])
    ax.set_xlabel(feat,fontsize=9)
    ax.set_ylabel("O'rtacha PD (%)")
    ax.set_title(f'PDP — {feat}',fontweight='bold',fontsize=10)

    # Rug plot
    rug = feat_orig.sample(min(200,len(feat_orig)),random_state=42)
    ax.plot(rug,[pdp_arr.min()*100-0.5]*len(rug),'|',
            color='gray',alpha=0.4,markersize=8)

fig.suptitle('Partial Dependence Plots — Top-4 Feature (GBM tuned)',
             fontsize=13,fontweight='bold')
plt.tight_layout()
plt.savefig('pdp_plots.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ pdp_plots.png saqlandi.')
print("✅ Qism 8 muvaffaqiyatli yakunlandi!")
