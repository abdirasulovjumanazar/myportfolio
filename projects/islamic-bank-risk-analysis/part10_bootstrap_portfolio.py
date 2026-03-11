
# ═══════════════════════════════════════════════════════════════
#  QISM 10 — Bootstrap CI + Portfel Optimizatsiyasi
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part9_threshold_eval import *

print("📈 10-Bo'lim: Bootstrap CI + Portfel Optimizatsiyasi boshlandi...")

# ── 10.1 Bootstrap 95% CI ────────────────────────────────────
print("Bootstrap 95% Ishonch Intervallari (n=1000)")
print('─'*55)
for name,res in ALL_MODELS.items():
    boot = []
    for _ in range(1000):
        idx = np.random.choice(len(y_te),len(y_te),replace=True)
        if y_te.values[idx].sum()>0:
            boot.append(roc_auc_score(y_te.values[idx],res['prob'][idx]))
    lo,hi = np.percentile(boot,[2.5,97.5])
    print(f'{name:<22} AUC={np.mean(boot):.4f}  95% CI=[{lo:.4f},{hi:.4f}]')

# ── 10.2 Portfel Optimizatsiyasi ─────────────────────────────
print('\n'+'═'*65)
print('PORTFEL OPTIMIZATSIYASI — Efficient Frontier')
print('═'*65)

mu_v  = np.array([df[df['xizmat_turi']==s]['foyda_stavkasi'].mean() for s in SVC_LIST])
sig_v = np.array([df[df['xizmat_turi']==s]['bozor_volatilligi'].mean() for s in SVC_LIST])
pd_v  = np.array([df[df['xizmat_turi']==s]['pd_qiymati'].mean() for s in SVC_LIST])
adj_mu = mu_v*(1-pd_v)

corr_m = np.array([[1.00,0.35,0.28,0.15],
                    [0.35,1.00,0.42,0.22],
                    [0.28,0.42,1.00,0.30],
                    [0.15,0.22,0.30,1.00]])
cov_m = np.outer(sig_v,sig_v)*corr_m
RF_RATE = 0.13

N_PORT = 15_000
p_ret,p_vol,p_sr,p_w = [],[],[],[]
for _ in range(N_PORT):
    w = np.random.dirichlet(np.ones(4))
    r = np.dot(w,adj_mu)
    v = np.sqrt(w@cov_m@w)
    s = (r-RF_RATE)/v if v>0 else 0
    p_ret.append(r); p_vol.append(v); p_sr.append(s); p_w.append(w)

p_ret=np.array(p_ret); p_vol=np.array(p_vol)
p_sr =np.array(p_sr);  p_w  =np.array(p_w)

msr_i = np.argmax(p_sr);  mvr_i = np.argmin(p_vol)
cur_w = np.array(list(SVC_DIST.values()))
cur_r = np.dot(cur_w,adj_mu); cur_v = np.sqrt(cur_w@cov_m@cur_w)

print('\nMax Sharpe portfeli:')
for s,w in zip(SVC_LIST,p_w[msr_i]):
    print(f'  {s:<13}: {w:.1%}')
print(f'  Return={p_ret[msr_i]:.2%}  Vol={p_vol[msr_i]:.2%}  SR={p_sr[msr_i]:.3f}')

# Vizualizatsiya
fig,axes = plt.subplots(1,2,figsize=(16,7))
ax1 = axes[0]
sc_plot = ax1.scatter(p_vol*100,p_ret*100,c=p_sr,cmap='RdYlGn',s=5,alpha=0.35,rasterized=True)
plt.colorbar(sc_plot,ax=ax1,label='Sharpe Ratio')
ax1.scatter(p_vol[msr_i]*100,p_ret[msr_i]*100,s=300,color='gold',
            marker='*',zorder=10,label='Max Sharpe ⭐',edgecolors='black')
ax1.scatter(p_vol[mvr_i]*100,p_ret[mvr_i]*100,s=200,color='blue',
            marker='^',zorder=10,label='Min Risk',edgecolors='white')
ax1.scatter(cur_v*100,cur_r*100,s=200,color='red',
            marker='D',zorder=10,label='Hozirgi',edgecolors='white')
for i,svc in enumerate(SVC_LIST):
    ax1.scatter(sig_v[i]*100,adj_mu[i]*100,s=120,color=PALETTE[i],
                marker='o',zorder=8,label=svc,edgecolors='white',lw=1)
ax1.set_xlabel('Volatillik (%)'); ax1.set_ylabel('Risk-Adj Daromad (%)')
ax1.set_title('Efficient Frontier'); ax1.legend(fontsize=8,loc='upper left')

ax2 = axes[1]
x = np.arange(4); w2=0.25
ax2.bar(x-w2,   cur_w*100,      w2,label='Hozirgi', color=PALETTE[4],alpha=0.85)
ax2.bar(x,       p_w[msr_i]*100,w2,label='Max Sharpe',color='gold',   alpha=0.85)
ax2.bar(x+w2,    p_w[mvr_i]*100,w2,label='Min Risk',  color='blue',   alpha=0.75)
ax2.set_xticks(x); ax2.set_xticklabels(SVC_LIST,fontsize=9)
ax2.set_title('Portfel Taqsimlash Taqqoslamasi'); ax2.set_ylabel('%')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('portfolio.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ portfolio.png saqlandi.')
print("✅ Qism 10 muvaffaqiyatli yakunlandi!")
