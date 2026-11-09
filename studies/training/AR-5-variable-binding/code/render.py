"""Technical research figures; primary and post-hoc panels remain separate."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SERIES=[('fixed',-.12,'#4169a1','s','Фиксированные имена'),('varied',.12,'#ba6125','o','Разнообразные имена')]
def save(fig,name):
    (ROOT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):
        p=ROOT/f'figures/{name}.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'AR-5'})
    rows=list(csv.DictReader((ROOT/'metrics/scores.csv').open(encoding='utf-8')))
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True);fig.subplots_adjust(top=.68,bottom=.18,wspace=.17)
    for ax,panel,title in zip(axes,['control','recombined','unseen'],['Контроль: a, b','Новые пары знакомых имён','Новые имена']):
        for condition,offset,color,marker,label in SERIES:
            values=[int(next(r for r in rows if r['seed']==seed and r['condition']==condition and r['panel']==panel)['functional_correct']) for seed in ['17','29','43']]
            ax.scatter([i+offset for i in range(3)],values,c=color,marker=marker,s=58,label=label,zorder=3)
        ax.set_title(title,fontsize=11,pad=12);ax.set_xticks(range(3),['17','29','43']);ax.set_ylim(-1.5,29);ax.set_yticks([0,9,18,27]);ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.17);ax.set_xlabel('Seed дообучения')
    axes[0].set_ylabel('Верных функций / 27')
    fig.suptitle('AR-5 · основная отложенная оценка',x=.055,y=.98,ha='left',fontsize=16)
    fig.text(.055,.86,'Один исходный checkpoint · 3 парных sampler seeds · каждая функция проверена на 25 входах',fontsize=10)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper left',bbox_to_anchor=(.05,.82),ncol=2,frameon=False)
    save(fig,'primary')
    rows=list(csv.DictReader((ROOT/'metrics/diagnostic.csv').open(encoding='utf-8')))
    fig,ax=plt.subplots(figsize=(9,4.8));fig.subplots_adjust(top=.68,bottom=.17)
    for condition,offset,color,marker,label in SERIES:
        values=[int(next(r for r in rows if r['seed']==seed and r['condition']==condition)['correct']) for seed in ['17','29','43']]
        pos=[i+offset for i in range(3)];ax.scatter(pos,values,c=color,marker=marker,s=64,label=label,zorder=3)
        for x,y in zip(pos,values):ax.annotate(str(y),(x,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_xticks(range(3),['17','29','43']);ax.set_xlabel('Seed дообучения');ax.set_ylabel('Верных функций / 18');ax.set_yticks([0,6,12,18]);ax.set_ylim(0,21);ax.set_xlim(-.5,2.5);ax.grid(axis='y',alpha=.17)
    fig.suptitle('AR-5 · дополнительная диагностика знакомых пар',x=.065,y=.98,ha='left',fontsize=15)
    fig.text(.065,.86,'Предложена после основных результатов · учебные пары имён, другие имена функций',fontsize=10)
    h,l=ax.get_legend_handles_labels();fig.legend(h,l,loc='upper left',bbox_to_anchor=(.055,.82),ncol=2,frameon=False)
    save(fig,'posthoc-seen-pairs')
if __name__=='__main__':main()
