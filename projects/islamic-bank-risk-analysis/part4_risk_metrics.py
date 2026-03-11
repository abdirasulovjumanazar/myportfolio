
# ═══════════════════════════════════════════════════════════════
#  QISM 4 — VaR / CVaR / Sharpe / Sortino / SRI / Basel EL
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part1_setup_dataset import *

print("📉 4-Bo'lim: Risk Ko'rsatkichlari boshlandi...")

def var_cvar(rets,alpha=0.95):
    v  = np.percentile(rets,(1-alpha)*100)
    cv = rets[rets<=v].mean() if (rets<=v).any() else v
    return v,cv

def sharpe(rets,rf=0.13/252):
    ex = rets-rf
    return np.sqrt(252)*ex.mean()/ex.std() if ex.std()>0 else 0

def sortino(rets,rf=0.13/252):
    ex = rets-rf
    ds = ex[ex<0].std()
    return np.sqrt(252)*ex.mean()/ds if ds>0 else 0

# ── 4.1 VaR tablosu ─────────────────────────────────────────
print('═'*90)
print(f'{"Xizmat":<13}',end='')
for cl in [90,95,99]:
    print(f'{"VaR("+str(cl)+"%) CVaR("+str(cl)+"%)":>24}',end='')
print(f'{"Sharpe":>9}{"Sortino":>9}')
print('─'*90)

SIM_RETS = {}
risk_tab  = []
for svc in SVC_LIST:
    sub = df[df['xizmat_turi']==svc]
    mu  = sub['foyda_stavkasi'].mean()/252
    sig = sub['bozor_volatilligi'].mean()/np.sqrt(252)
    sim = np.random.normal(mu,sig,50_000)
    SIM_RETS[svc] = sim
    row = [svc]
    line = f'{svc:<13}'
    for cl in [0.90,0.95,0.99]:
        v,cv = var_cvar(sim,cl)
        line += f'{v:>12.5f}{cv:>12.5f}'
        row += [round(v,5),round(cv,5)]
    sh = sharpe(sim); so = sortino(sim)
    line += f'{sh:>9.3f}{so:>9.3f}'
    print(line)
    risk_tab.append(row)
print('═'*90)

# ── 4.2 SRI ─────────────────────────────────────────────────
print('\n'+'═'*65)
print('SHARIA RISK INDEKSI (SRI) — AAOIFI asosida')
print('W: PD=35% | Bozor=25% | Sharia NC=25% | Likvidlik=15%')
print('═'*65)
W = {'pd':0.35,'mkt':0.25,'sh':0.25,'liq':0.15}
SRI_RES = {}
for svc in SVC_LIST:
    sub = df[df['xizmat_turi']==svc]
    pd_r = sub['pd_qiymati'].mean()
    mk_r = sub['bozor_volatilligi'].mean()
    sh_r = (1-sub['sharia_audit'].mean()) + sub['gharar_darajasi'].mean()*0.5
    lq_r = 1-sub['likvidlik'].mean()
    sri  = W['pd']*pd_r + W['mkt']*mk_r + W['sh']*sh_r + W['liq']*lq_r
    g    = ('A ✅' if sri<0.08 else ('B 🟡' if sri<0.14 else ('C 🟠' if sri<0.22 else 'D 🔴')))
    SRI_RES[svc] = {'sri':sri,'grade':g}
    print(f'{svc:<13} SRI={sri:.5f}  Daraja={g}')
    print(f'  PD={pd_r:.4f} | Bozor={mk_r:.4f} | ShariaNc={sh_r:.4f} | Liq={lq_r:.4f}')

# ── 4.3 Basel III EL/UL ─────────────────────────────────────
print('\n'+'═'*65)
print('BASEL III (Islomiy) — EL · UL · Kapital Zaxirasi')
print('─'*65)
print(f'{"Xizmat":<13}{"EL (mln)":>12}{"UL (mln)":>12}{"Kapital%":>10}')
print('─'*65)
for svc in SVC_LIST:
    sub = df[df['xizmat_turi']==svc]
    EL  = (sub['pd_qiymati']*sub['ead']*sub['lgd']).sum()/1e6
    loss= sub['pd_qiymati']*sub['ead']*sub['lgd']
    UL  = loss.std()*2.33/1e6
    cap = UL/(sub['ead'].sum()/1e6)*100
    print(f'{svc:<13}{EL:>12.2f}{UL:>12.2f}{cap:>10.2f}%')
print('═'*65)
print("✅ Qism 4 muvaffaqiyatli yakunlandi!")
