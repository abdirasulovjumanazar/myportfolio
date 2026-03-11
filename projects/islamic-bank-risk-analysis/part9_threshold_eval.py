
# ═══════════════════════════════════════════════════════════════
#  QISM 9 — Threshold Optimization + Model Baholash Dashboard
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part7_ml_models import *

print("🎯 9-Bo'lim: Threshold Optimization + Model Baholash boshlandi...")

# ── 9.1 Threshold Optimization ───────────────────────────────
ens_prob = ALL_MODELS['Ensemble']['prob']
thrs = np.arange(0.10,0.90,0.02)
thr_rows = []
for t in thrs:
    yp = (ens_prob>=t).astype(int)
    thr_rows.append({'thr':t,
        'f1':   f1_score(y_te,yp,zero_division=0),
        'prec': precision_score(y_te,yp,zero_division=0),
        'rec':  recall_score(y_te,yp,zero_division=0),
        'acc':  accuracy_score(y_te,yp)})
thr_df = pd.DataFrame(thr_rows)
BEST_THR = thr_df.loc[thr_df['f1'].idxmax(),'thr']
print(f'Optimal threshold: {BEST_THR:.2f}  '
      f'F1={thr_df.loc[thr_df["f1"].idxmax(),"f1"]:.4f}')

# ── 9.2 Katta Dashboard ──────────────────────────────────────
MCOLORS = {'Logistic Regression':PALETTE[0],'Random Forest':PALETTE[1],
            'GBM (Tuned)':PALETTE[2],'Ensemble':PALETTE[3]}

fig = plt.figure(figsize=(20,15))
gs  = gridspec.GridSpec(3,4,hspace=0.50,wspace=0.40)

# ROC
ax_roc = fig.add_subplot(gs[0,0:2])
ax_roc.plot([0,1],[0,1],'k--',alpha=0.4,label='Random')
for name,res in ALL_MODELS.items():
    fpr,tpr,_ = roc_curve(y_te,res['prob'])
    lw = 2.5 if name=='Ensemble' else 1.5
    ax_roc.plot(fpr,tpr,color=MCOLORS.get(name,'gray'),lw=lw,
                label=f"{name} (AUC={res['auc']:.3f})")
ax_roc.set_title('ROC Egri Chiziqlari'); ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=8)

# Precision-Recall
ax_pr = fig.add_subplot(gs[0,2:4])
for name,res in ALL_MODELS.items():
    pr,re,_ = precision_recall_curve(y_te,res['prob'])
    ap = average_precision_score(y_te,res['prob'])
    ax_pr.plot(re,pr,color=MCOLORS.get(name,'gray'),lw=1.5,
               label=f'{name} AP={ap:.3f}')
ax_pr.axhline(y_te.mean(),color='gray',ls='--',lw=1,label='Baseline')
ax_pr.set_title('Precision-Recall'); ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.legend(fontsize=8)

# Confusion Matrices
for col_i,(name,res) in enumerate(list(ALL_MODELS.items())):
    ax_cm = fig.add_subplot(gs[1,col_i])
    yp = (res['prob']>=BEST_THR).astype(int)
    cm = confusion_matrix(y_te,yp)
    sns.heatmap(cm,annot=True,fmt='d',ax=ax_cm,cmap='Blues',
                linewidths=1.5,
                xticklabels=['Normal','Default'],
                yticklabels=['Normal','Default'],
                annot_kws={'size':13})
    tn,fp,fn,tp = cm.ravel()
    sens = tp/(tp+fn) if tp+fn>0 else 0
    spec = tn/(tn+fp) if tn+fp>0 else 0
    ax_cm.set_title(f'{name}\nSens={sens:.2f} Spec={spec:.2f}',
                    fontweight='bold',fontsize=8)

# Threshold Plot
ax_thr = fig.add_subplot(gs[2,0:2])
ax_thr.plot(thr_df['thr'],thr_df['f1'],  color=PALETTE[2],lw=2,label='F1')
ax_thr.plot(thr_df['thr'],thr_df['prec'],color=PALETTE[0],lw=1.5,label='Precision')
ax_thr.plot(thr_df['thr'],thr_df['rec'], color=PALETTE[1],lw=1.5,label='Recall')
ax_thr.axvline(BEST_THR,color='black',ls='--',lw=1.5,label=f'Opt={BEST_THR:.2f}')
ax_thr.set_title('Threshold Optimization'); ax_thr.set_xlabel('Threshold')
ax_thr.legend(); ax_thr.set_xlim(0.1,0.9)

# Calibration
ax_cal = fig.add_subplot(gs[2,2:4])
ax_cal.plot([0,1],[0,1],'k--',label='Perfect')
for name,res in ALL_MODELS.items():
    fp2,mp2 = calibration_curve(y_te,res['prob'],n_bins=10)
    ax_cal.plot(mp2,fp2,'s-',color=MCOLORS.get(name,'gray'),lw=1.5,
                markersize=4,label=name)
ax_cal.set_title('Kalibrasiya'); ax_cal.set_xlabel('Bashorat'); ax_cal.set_ylabel('Haqiqiy')
ax_cal.legend(fontsize=8)

fig.suptitle("To'liq Model Baholash Dashboard",fontsize=15,fontweight='bold')
plt.savefig('model_evaluation.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ model_evaluation.png saqlandi.')
print("✅ Qism 9 muvaffaqiyatli yakunlandi!")
