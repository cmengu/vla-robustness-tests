# Renders fig1-3 from the numbers in RESULTS.md. Run: python3 -I figures/make_figures.py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os; OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
BLUE,ORANGE,AQUA,GRAY,INK,MUTED="#2a78d6","#eb6834","#1baf7a","#9a9890","#2b2a27","#6f6e68"
plt.rcParams.update({"font.size":10,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
  "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","axes.facecolor":"white","text.color":INK})
def grid(ax,axis="y"): ax.grid(axis=axis,color="#e6e5e0",lw=0.8); ax.set_axisbelow(True)
# ---- Figure 1: Test 1 delay
d=[0,2,5,10,20]; fw=[94,94,85,80,41]; a1=[99,93,95,81,50]; a10=[97,96,89,81,34]
fig,ax=plt.subplots(figsize=(6.4,3.6),dpi=160)
ax.plot(d,a1,color=BLUE,lw=2,marker="o",ms=5,label="Alpha-S, re-plan every step")
ax.plot(d,a10,color=BLUE,lw=2,ls="--",marker="s",ms=5,label="Alpha-S, re-plan every 10 steps")
ax.plot(d,fw,color=ORANGE,lw=2,marker="o",ms=5,label="Fast-WAM")
for x,y,c in ((20,50,BLUE),(20,34,BLUE),(20,41,ORANGE)): ax.annotate(f"{y}",(x,y),xytext=(6,-3),textcoords="offset points",color=INK,fontsize=9)
ax.set_xticks(d); ax.set_xlabel("Delay d (control steps of 0.1 s)"); ax.set_ylabel("Success (%)"); ax.set_ylim(0,105); ax.set_xlim(-0.5,22)
ax.set_title("Test 1: success under delayed actions (libero_object, 100 episodes per point)",fontsize=10,loc="left")
grid(ax); ax.legend(frameon=False,fontsize=9,loc="lower left"); fig.tight_layout(); fig.savefig(OUT+"fig1_delay.png"); plt.close(fig)
# ---- Figure 2: Test 3 factors
fac=["Camera viewpoint","Robot start pose","Table layout","Sensor noise","Extra objects","Language rewording","Lighting"]
A=[75,76,89,69,78,87,88]; F=[25,41,55,39,56,68,79]; wins=["54 / 4","39 / 4","38 / 4","39 / 9","25 / 3","23 / 4","13 / 4"]
y=np.arange(len(fac))[::-1]; h=0.36
fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=160)
ax.barh(y+h/2,A,h-0.04,color=BLUE,label="Alpha-S"); ax.barh(y-h/2,F,h-0.04,color=ORANGE,label="Fast-WAM")
for yi,a,f,w in zip(y,A,F,wins):
    ax.text(a+1,yi+h/2,f"{a}",va="center",fontsize=8.5,color=INK); ax.text(f+1,yi-h/2,f"{f}",va="center",fontsize=8.5,color=INK)
    ax.text(103,yi,w,va="center",fontsize=8.5,color=MUTED)
ax.text(103,len(fac)-0.35,"A>F / F>A",fontsize=8.5,color=MUTED,va="center")
ax.set_yticks(y); ax.set_yticklabels(fac); ax.set_xlim(0,118); ax.set_xticks([0,25,50,75,100]); ax.set_xlabel("Success (%) on 100 paired cases per factor")
ax.set_title("Test 3: success by LIBERO-Plus perturbation factor (700 paired cases)",fontsize=10,loc="left")
grid(ax,"x"); ax.legend(frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(0.45,-0.14),ncol=2); fig.tight_layout(); fig.savefig(OUT+"fig2_factors.png"); plt.close(fig)
# ---- Figure 3: Test 4 arms + probe
fac4=["Camera viewpoint","Robot start pose","Table layout","Sensor noise","Extra objects","Language rewording","Lighting","All 700"]
a=[73,68,92,69,80,89,93,80.6]; b=[74,70,94,63,82,85,91,79.9]; c=[72,72,97,68,79,88,91,81.0]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,4.6),dpi=160,gridspec_kw={"width_ratios":[2.2,1]})
y=np.arange(len(fac4))[::-1]; h=0.26
ax1.barh(y+h,a,h-0.03,color=BLUE,label="A: plain fine-tune"); ax1.barh(y,b,h-0.03,color=ORANGE,label="B: + future head"); ax1.barh(y-h,c,h-0.03,color=AQUA,label="C: + current-frame head")
for yi,va,vb,vc in zip(y,a,b,c):
    for off,v in ((h,va),(0,vb),(-h,vc)): ax1.text(v+1,yi+off,f"{v:g}",va="center",fontsize=7.5,color=INK)
ax1.set_yticks(y); ax1.set_yticklabels(fac4); ax1.set_xlim(0,112); ax1.set_xticks([0,25,50,75,100]); ax1.set_xlabel("Success (%), same 700 cases; 100 per factor")
ax1.set_title("Test 4: three arms of Alpha-S by factor",fontsize=10,loc="left"); grid(ax1,"x"); ax1.legend(frameon=False,fontsize=8.5,loc="upper center",bbox_to_anchor=(0.45,-0.12),ncol=3)
names=["Stock","A","B","C"]; r2=[0.21,0.21,0.33,0.27]; cols=[GRAY,BLUE,ORANGE,AQUA]
ax2.bar(names,r2,0.55,color=cols)
for i,v in enumerate(r2): ax2.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=8.5,color=INK)
ax2.set_ylim(0,0.4); ax2.set_ylabel("Held-out R², linear probe"); ax2.set_title("Probe: future information\nat the layer the head read",fontsize=10,loc="left"); grid(ax2)
fig.tight_layout(); fig.savefig(OUT+"fig3_arms.png"); plt.close(fig)
print("ok")
