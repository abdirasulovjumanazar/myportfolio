
# ═══════════════════════════════════════════════════════════════
#  QISM 5 — Monte Carlo GBM + Stress Testing
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part1_setup_dataset import *

print("🎲 5-Bo'lim: Monte Carlo + Stress Testing boshlandi...")

N_SIM = 20_000; HOR = 252

MC = {}
for svc in SVC_LIST:
    sub = df[df['xizmat_turi']==svc]
    mu  = sub['foyda_stavkasi'].mean()
    sig = sub['bozor_volatilligi'].mean()
    S0  = sub['ead'].mean()
    dt  = 1/HOR
    dW  = np.random.normal(0,np.sqrt(dt),(N_SIM,HOR))
    ret = (mu-0.5*sig**2)*dt + sig*dW
    paths = S0*np.exp(np.cumsum(ret,axis=1))
    fin   = paths[:,-1]
    MC[svc] = {
        'paths':paths[:100,:],'final':fin,'S0':S0,
        'mean':fin.mean(),'std':fin.std(),
        'var95':np.percentile(fin,5),
        'var99':np.percentile(fin,1),
        'prob_loss':(fin<S0).mean()
    }

# Stress scenariylari
STRESS = {
    'Asosiy':           {'pm':1.0, 'vm':1.0, 'rd':0.000},
    'Yengil Stress':    {'pm':1.5, 'vm':1.3, 'rd':0.020},
    "O'rta Stress":    {'pm':2.5, 'vm':1.8, 'rd':0.050},
    "Og'ir Stress":    {'pm':4.0, 'vm':2.5, 'rd':0.100},
    'COVID-19 analog':  {'pm':5.5, 'vm':3.5, 'rd':0.080},
    'Valyuta inqirozi': {'pm':3.0, 'vm':4.0, 'rd':0.060},
}
stress_rows = []
for sc,par in STRESS.items():
    row={'Senariy':sc}
    tot=0
    for svc in SVC_LIST:
        sub = df[df['xizmat_turi']==svc]
        spd = np.clip(sub['pd_qiymati']*par['pm'],0,1)
        sea = sub['ead']*(1+par['rd'])
        el  = (spd*sea*sub['lgd']).sum()/1e6
        row[svc]=round(el,2); tot+=el
    row['JAMI']=round(tot,2); stress_rows.append(row)
stress_df = pd.DataFrame(stress_rows).set_index('Senariy')
print('Monte Carlo va Stress Testing bajarildi.')
print(stress_df.to_string())

# ── Vizualizatsiya ──
fig,axes = plt.subplots(2,2,figsize=(16,11))
days = np.arange(HOR)
for idx,(svc,color) in enumerate(zip(SVC_LIST,PALETTE)):
    ax = axes[idx//2,idx%2]
    ps = MC[svc]['paths']; S0=MC[svc]['S0']
    for path in ps[:60]:
        ax.plot(days,path/S0,alpha=0.07,color=color,lw=0.7)
    p5  = np.percentile(ps,5, axis=0)/S0
    p50 = np.percentile(ps,50,axis=0)/S0
    p95 = np.percentile(ps,95,axis=0)/S0
    ax.plot(days,p50,color='navy',lw=2,  label='P50')
    ax.plot(days,p5, color='red', lw=1.5,ls='--',label='P5')
    ax.plot(days,p95,color='green',lw=1.5,ls='--',label='P95')
    ax.fill_between(days,p5,p95,alpha=0.1,color=color)
    ax.axhline(1,color='black',lw=1,ls=':')
    r=MC[svc]
    ax.set_title(f"{svc}\nZarar ehtimoli={r['prob_loss']:.1%}  VaR99={r['var99']/S0:.3f}")
    ax.set_xlabel('Kun'); ax.set_ylabel('Normallashtirilgan')
    ax.legend(fontsize=8)
fig.suptitle(f'Monte Carlo GBM ({N_SIM:,} Senariy)',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig('monte_carlo.png',bbox_inches='tight',dpi=150)
plt.close()

fig2,ax2 = plt.subplots(figsize=(12,5))
stress_df[SVC_LIST].plot(kind='bar',ax=ax2,color=PALETTE[:4],edgecolor='white')
ax2.set_title("Stress Testing — Senariy bo'yicha EL (mln UZS)",fontweight='bold')
ax2.set_ylabel('EL (mln UZS)'); ax2.tick_params(axis='x',rotation=20)
plt.tight_layout()
plt.savefig('stress_testing.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ monte_carlo.png, stress_testing.png saqlandi.')
print("✅ Qism 5 muvaffaqiyatli yakunlandi!")
