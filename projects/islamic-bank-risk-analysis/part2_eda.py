
# ═══════════════════════════════════════════════════════════════
#  QISM 2 — Kengaytirilgan EDA (8 grafik bir sahifada)
# ═══════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from part1_setup_dataset import *

print("📊 2-Bo'lim: Kengaytirilgan EDA boshlandi...")

fig = plt.figure(figsize=(20,15))
gs  = gridspec.GridSpec(3,4,hspace=0.50,wspace=0.38)

# 1. Pie
ax = fig.add_subplot(gs[0,0])
cnt = df['xizmat_turi'].value_counts()
ax.pie(cnt,labels=cnt.index,autopct='%1.1f%%',colors=PALETTE,
       startangle=90,wedgeprops={'linewidth':2,'edgecolor':'white'})
ax.set_title('Xizmat Turlari Ulushi')

# 2. Default nisbati
ax = fig.add_subplot(gs[0,1])
dr = df.groupby('xizmat_turi')['default_holati'].mean().sort_values(ascending=False)
bars = ax.bar(dr.index,dr.values*100,color=PALETTE[:4],edgecolor='white',lw=1.5)
for b,v in zip(bars,dr.values):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.3,f'{v:.1%}',
            ha='center',fontsize=9,fontweight='bold')
ax.set_title('Default Nisbati'); ax.set_ylabel('%')

# 3. Violin — PD
ax = fig.add_subplot(gs[0,2])
data_v = [df[df['xizmat_turi']==s]['pd_qiymati'].values for s in SVC_LIST]
parts = ax.violinplot(data_v,showmedians=True,showextrema=True)
for i,pc in enumerate(parts['bodies']):
    pc.set_facecolor(PALETTE[i]); pc.set_alpha(0.75)
ax.set_xticks([1,2,3,4]); ax.set_xticklabels(SVC_LIST,fontsize=8)
ax.set_title('PD Violin'); ax.set_ylabel('Default Ehtimoli')

# 4. Box — Kredit ball
ax = fig.add_subplot(gs[0,3])
grp = [df[df['xizmat_turi']==s]['kredit_ball'].values for s in SVC_LIST]
bp = ax.boxplot(grp,patch_artist=True,
                boxprops=dict(facecolor='#D6EAF8'),
                medianprops=dict(color='navy',lw=2))
ax.set_xticks([1,2,3,4]); ax.set_xticklabels(SVC_LIST,fontsize=8,rotation=15)
ax.set_title('Kredit Ball Taqsimlashi'); ax.set_ylabel('Kredit Ball')

# 5. Scatter LTV vs PD
ax = fig.add_subplot(gs[1,0])
for i,svc in enumerate(SVC_LIST):
    sub = df[df['xizmat_turi']==svc]
    ax.scatter(sub['ltv_nisbati'],sub['pd_qiymati'],s=10,alpha=0.3,
               color=PALETTE[i],label=svc)
z = np.polyfit(df['ltv_nisbati'],df['pd_qiymati'],1)
xl = np.linspace(0.1,0.95,100)
ax.plot(xl,np.poly1d(z)(xl),'k--',lw=1.5,label='Trend')
ax.set_title('LTV va PD'); ax.set_xlabel('LTV'); ax.set_ylabel('PD')
ax.legend(fontsize=7,markerscale=2)

# 6. Stacked bar — Risk darajasi
ax = fig.add_subplot(gs[1,1])
rct = df.groupby(['xizmat_turi','risk_darajasi']).size().unstack(fill_value=0)
rct_pct = rct.div(rct.sum(1),0)*100
rct_pct.plot(kind='bar',stacked=True,ax=ax,
             color=['#1E8449','#F39C12','#E74C3C','#8E44AD'],
             edgecolor='white',lw=0.4)
ax.set_title('Risk Darajasi (%)'); ax.set_ylabel('%'); ax.set_xlabel('')
ax.tick_params(axis='x',rotation=20); ax.legend(fontsize=7)

# 7. Sharia vs PD
ax = fig.add_subplot(gs[1,2])
bins = pd.cut(df['sharia_audit'],6)
spd  = df.groupby(bins, observed=True)['pd_qiymati'].mean()
ax.bar(range(len(spd)),spd.values*100,color=PALETTE[2],edgecolor='white')
ax.set_xticks(range(len(spd)))
ax.set_xticklabels([f'{i.left:.2f}' for i in spd.index],fontsize=8,rotation=30)
ax.set_title("Sharia Audit va O'rt. PD"); ax.set_ylabel('PD (%)')

# 8. Kutilgan zarar
ax = fig.add_subplot(gs[1,3])
el = df.groupby('xizmat_turi')['kutilgan_zarar'].mean()/1e6
ax.barh(el.sort_values().index,el.sort_values().values,
        color=PALETTE[:4],edgecolor='white')
ax.set_title("O'rt. Kutilgan Zarar"); ax.set_xlabel('mln UZS')

# 9. Korrelyatsiya heatmap (pastki qator)
ax = fig.add_subplot(gs[2,:])
nc = ['kredit_ball','ltv_nisbati','foyda_stavkasi','bozor_volatilligi',
      'likvidlik','sharia_audit','gharar_darajasi','qarz_xizmat_nisbati',
      'inflyatsiya','valyuta_tebranishi','leverage','pd_qiymati','default_holati']
corr = df[nc].corr()
mask = np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr,mask=mask,ax=ax,annot=True,fmt='.2f',
            cmap='RdYlGn_r',center=0,vmin=-1,vmax=1,
            linewidths=0.4,annot_kws={'size':7.5},
            xticklabels=[c[:12] for c in nc],
            yticklabels=[c[:15] for c in nc])
ax.set_title('Korrelyatsiya Matritsasi',fontsize=12)

fig.suptitle('Islomiy Bank — Kengaytirilgan EDA',fontsize=15,fontweight='bold')
plt.savefig('eda.png',bbox_inches='tight',dpi=150)
plt.close()
print('✅ EDA saqlandi: eda.png')
print("✅ Qism 2 muvaffaqiyatli yakunlandi!")
