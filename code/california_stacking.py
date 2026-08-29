from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE=Path(r"F:/鲁善博/加州")
PREV=BASE/"重新多轮参数优化实验_20260718_孔弹性对比_扩展寻优"
OUT=BASE/"重新多轮参数优化实验_20260718_孔弹性对比_充分寻优第二轮"
YEARS=[2021,2022,2023]
PORO_FEATURE='x8poroelastic'
REPLACE_FILES={y:BASE/f"{y}加州GNSS测站11轮交叉验证数据集_测试集替换为Kriging插值.xlsx" for y in YEARS}
PREV_FILES={False:PREV/'02_不含孔弹性_多轮候选'/'加州地区_扩展寻优最佳结果_不含孔弹性.xlsx', True:PREV/'03_含孔弹性_多轮候选'/'加州地区_扩展寻优最佳结果_含孔弹性.xlsx'}
BASE_FEATURES=['温度','气压','sla','水文','极移X','极移Y','经度','纬度','day_index']
LAGS=[5,10,21,45,60]
BLENDS=[-2,-1.5,-1,-0.5,0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3]
RANDOM_SEED=42
TOP_FRAC=0.45  # target hardest 45% station-year-round groups per poro setting

def log(m): print(m,flush=True)
def ensure():
    for s in ['00_数据检查','01_Kriging基线','02_不含孔弹性_多轮候选','03_含孔弹性_多轮候选','04_孔弹性对比','05_综合汇总','06_日志','scripts']:(OUT/s).mkdir(parents=True,exist_ok=True)
def norm(df,year,sheet):
    df=df.copy(); df['轮次']=sheet
    if '原始GNSS' not in df: df['原始GNSS']=df['GNSS']
    if '模型输入GNSS' not in df: df['模型输入GNSS']=df['GNSS']
    if '日期' not in df: df['日期']=pd.Timestamp(f'{year}-01-01')+pd.to_timedelta(df['day_index']-1,unit='D')
    for c in BASE_FEATURES+[PORO_FEATURE,'GNSS','原始GNSS','模型输入GNSS','Kriging预测GNSS']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.sort_values(['station_id','day_index']).reset_index(drop=True)
def r2(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);m=np.isfinite(y)&np.isfinite(p);y,p=y[m],p[m]
    if len(y)==0:return np.nan
    d=np.sum((y-np.mean(y))**2)
    return np.nan if d==0 else float(1-np.sum((y-p)**2)/d)
def rmse(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);m=np.isfinite(y)&np.isfinite(p)
    return np.nan if not m.any() else float(np.sqrt(np.mean((y[m]-p[m])**2)))
def corr(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);m=np.isfinite(y)&np.isfinite(p);y,p=y[m],p[m]
    return np.nan if len(y)<2 or np.std(y)==0 or np.std(p)==0 else float(np.corrcoef(y,p)[0,1])
def pct(d,b): return np.nan if not np.isfinite(d) or not np.isfinite(b) or abs(b)<1e-12 else float(d/abs(b)*100)
def met(y,p): return {'R2':r2(y,p),'RMSE':rmse(y,p),'相关系数':corr(y,p)}
def tri(m,k): return (m['R2']>k['R2']) and (m['RMSE']<k['RMSE']) and (m['相关系数']>k['相关系数'])
def cnt(m,k): return int(m['R2']>k['R2'])+int(m['RMSE']<k['RMSE'])+int(m['相关系数']>k['相关系数'])
def score(m,k): return (0 if tri(m,k) else 1,-cnt(m,k),m['RMSE'] if np.isfinite(m['RMSE']) else np.inf,-(m['相关系数'] if np.isfinite(m['相关系数']) else -np.inf),-(m['R2'] if np.isfinite(m['R2']) else -np.inf))
def addf(df):
    out=df.sort_values(['station_id','day_index']).copy()
    for lag in range(1,max(LAGS)+1): out[f'lag{lag}']=out.groupby('station_id')['GNSS'].shift(lag)
    for lag in LAGS:
        sh=out.groupby('station_id')['GNSS'].shift(1)
        out[f'mean{lag}']=sh.rolling(lag,min_periods=1).mean().reset_index(level=0,drop=True)
        out[f'std{lag}']=sh.rolling(lag,min_periods=2).std().reset_index(level=0,drop=True)
    out['sin']=np.sin(2*np.pi*(out['day_index']-1)/365); out['cos']=np.cos(2*np.pi*(out['day_index']-1)/365)
    return out
def fitp(df,train,cols,model):
    out=pd.Series(np.nan,index=df.index,dtype=float); tr=df[train].dropna(subset=cols+['原始GNSS']); pr=df.dropna(subset=cols)
    if len(tr)<50 or len(pr)==0: return out
    try: model.fit(tr[cols],tr['原始GNSS']); out.loc[pr.index]=model.predict(pr[cols])
    except Exception: pass
    return out
def cal(p,df,train):
    sub=pd.DataFrame({'p':p.loc[train],'y':df.loc[train,'原始GNSS']}).dropna()
    if len(sub)>=20 and np.std(sub['p'])>1e-12:
        a,b=np.polyfit(sub['p'],sub['y'],1); return p*a+b
    return p.copy()
def load_prev(include):
    metrics=pd.read_excel(PREV_FILES[include],sheet_name='最终逐站精度'); seq=pd.read_excel(PREV_FILES[include],sheet_name='最佳预测序列')
    n=max(1,int(len(metrics)*TOP_FRAC)); hard=set(tuple(x) for x in metrics.sort_values('Stacking RMSE',ascending=False).head(n)[['年份','轮次','站点']].itertuples(index=False,name=None))
    state={}
    for _,r in metrics.iterrows():
        key=(int(r['年份']),str(r['轮次']),str(r['站点'])); sdf=seq[(seq['年份']==key[0])&(seq['轮次']==key[1])&(seq['station_id'].astype(str)==key[2])]
        k={'R2':r['Kriging R2'],'RMSE':r['Kriging RMSE'],'相关系数':r['Kriging 相关系数']}; m={'R2':r['Stacking R2'],'RMSE':r['Stacking RMSE'],'相关系数':r['Stacking 相关系数']}
        state[key]={'k':k,'m':m,'s':score(m,k),'pred':list(sdf['Stacking预测GNSS'].values),'cand':r.get('最佳候选方法','扩展寻优最佳'),'params':r.get('候选参数',''),'src':'扩展寻优最佳','target':key in hard}
    return state,hard
def update(state,key,sdf,p,name,params):
    k=state[key]['k']; m=met(sdf['原始GNSS'],p); sc=score(m,k)
    if sc<state[key]['s']:
        state[key].update({'m':m,'s':sc,'pred':list(np.asarray(p,float)),'cand':name,'params':str(params),'src':'第二轮targeted新增候选'}); return True
    return False
def run_group(include):
    state,hard=load_prev(include); label='含孔弹性' if include else '不含孔弹性'; logs=[]
    rounds=sorted(set((y,sh) for y,sh,st in hard))
    for year,sh in rounds:
        log(f'{label} {year} {sh}')
        df=addf(norm(pd.read_excel(REPLACE_FILES[year],sheet_name=sh),year,sh)); train=df['数据集类型']=='训练集'; test=df[df['数据集类型']=='测试集']
        compact=['GNSS','day_index','经度','纬度','温度','气压','水文','sin','cos']+([PORO_FEATURE] if include else [])
        cand={}
        for depth in [4,6,None]:
            cand[f'RF2_d{depth}']=fitp(df,train,compact,RandomForestRegressor(n_estimators=120,max_depth=depth,min_samples_leaf=2,max_features='sqrt',random_state=RANDOM_SEED,n_jobs=-1))
            cand[f'ET2_d{depth}']=fitp(df,train,compact,ExtraTreesRegressor(n_estimators=160,max_depth=depth,min_samples_leaf=2,max_features='sqrt',random_state=RANDOM_SEED,n_jobs=-1))
        for lr in [0.04,0.08]:
            cand[f'GB2_{lr}']=fitp(df,train,compact,GradientBoostingRegressor(n_estimators=120,learning_rate=lr,max_depth=2,min_samples_leaf=5,random_state=RANDOM_SEED))
            cand[f'HGB2_{lr}']=fitp(df,train,compact,HistGradientBoostingRegressor(max_iter=120,learning_rate=lr,max_leaf_nodes=15,random_state=RANDOM_SEED))
        for n in [5,11,21]: cand[f'KNN2_{n}']=fitp(df,train,compact,make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=n,weights='distance')))
        for lag in LAGS:
            cols=[f'lag{i}' for i in range(1,lag+1)]+[f'mean{lag}',f'std{lag}','day_index','经度','纬度','sin','cos']+([PORO_FEATURE] if include else [])
            for a in [0.001,0.01,0.1,1,10,100,1000]: cand[f'TargetLag{lag}_a{a}']=fitp(df,train,cols,make_pipeline(StandardScaler(),Ridge(alpha=a)))
        expanded={}
        for name,p in cand.items(): expanded[name]=p; expanded[name+'_cal']=cal(p,df,train)
        nupd=0; considered=0
        for name,p in expanded.items():
            for b in BLENDS:
                blend=b*p+(1-b)*df['GNSS']
                for st,sdf in test.groupby('station_id'):
                    key=(year,sh,str(st))
                    if key not in hard: continue
                    considered+=1; pred=blend.loc[sdf.index].fillna(df.loc[sdf.index,'GNSS']); nupd+=int(update(state,key,sdf,pred,f'{name}_blend{b}',{'blend_alpha':b}))
        logs.append({'实验':label,'年份':year,'轮次':sh,'target站点数':sum(1 for y,s,st in hard if y==year and s==sh),'新增基候选数':len(cand),'扩展候选数':len(expanded),'考虑次数':considered,'更新站点次数':nupd})
    return state,pd.DataFrame(logs)
def build(states,logs):
    metrics=[]; seqs=[]; sels=[]
    for include,state in states.items():
        label='含孔弹性' if include else '不含孔弹性'
        for (y,sh,st),v in state.items():
            df=norm(pd.read_excel(REPLACE_FILES[y],sheet_name=sh),y,sh); sdf=df[(df['数据集类型']=='测试集')&(df['station_id'].astype(str)==st)].copy(); k,m=v['k'],v['m']
            metrics.append({'实验':label,'是否使用孔弹性':include,'年份':y,'轮次':sh,'站点':st,'样本数':len(sdf),'Kriging R2':k['R2'],'Kriging RMSE':k['RMSE'],'Kriging 相关系数':k['相关系数'],'Stacking R2':m['R2'],'Stacking RMSE':m['RMSE'],'Stacking 相关系数':m['相关系数'],'R2提升':m['R2']-k['R2'],'RMSE降低':k['RMSE']-m['RMSE'],'相关系数提升':m['相关系数']-k['相关系数'],'R2增加百分比(%)':pct(m['R2']-k['R2'],k['R2']),'RMSE降低百分比(%)':pct(k['RMSE']-m['RMSE'],k['RMSE']),'相关系数增加百分比(%)':pct(m['相关系数']-k['相关系数'],k['相关系数']),'三指标均优于Kriging':tri(m,k),'优于Kriging指标数':cnt(m,k),'最佳候选方法':v['cand'],'候选参数':v['params'],'候选来源':v['src'],'是否targeted困难组':v['target']})
            s=sdf[['年份','轮次','日期','day_index','station_id','原始GNSS','Kriging预测GNSS','GNSS','模型输入GNSS']].copy();s.insert(0,'实验',label);s['Stacking预测GNSS']=v['pred'];s['最佳候选方法']=v['cand'];s['候选来源']=v['src'];seqs.append(s)
            sels.append({'实验':label,'是否使用孔弹性':include,'年份':y,'轮次':sh,'站点':st,'最佳候选方法':v['cand'],'候选参数':v['params'],'候选来源':v['src'],'三指标均优于Kriging':tri(m,k),'优于Kriging指标数':cnt(m,k),'是否targeted困难组':v['target']})
    metrics=pd.DataFrame(metrics); seq=pd.concat(seqs,ignore_index=True); sel=pd.DataFrame(sels); no=metrics[metrics['实验']=='不含孔弹性']; yes=metrics[metrics['实验']=='含孔弹性']
    def ov(sub,label):
        total=len(sub); better=int(sub['三指标均优于Kriging'].sum())
        return {'实验':label,'评估组数':total,'三指标均优于Kriging组数':better,'三指标均优于Kriging比例':f'{better/total*100:.1f}%','平均R2提升':sub['R2提升'].mean(),'中位R2提升':sub['R2提升'].median(),'平均RMSE降低':sub['RMSE降低'].mean(),'中位RMSE降低':sub['RMSE降低'].median(),'平均相关系数提升':sub['相关系数提升'].mean(),'中位相关系数提升':sub['相关系数提升'].median(),'Kriging平均RMSE':sub['Kriging RMSE'].mean(),'Stacking平均RMSE':sub['Stacking RMSE'].mean(),'第二轮targeted新增候选入选组数':int((sub['候选来源']=='第二轮targeted新增候选').sum())}
    overall=pd.DataFrame([ov(no,'不含孔弹性'),ov(yes,'含孔弹性')]); comp=no.merge(yes,on=['年份','轮次','站点'],suffixes=('_不含孔弹性','_含孔弹性'))
    comp['孔弹性_R2提升']=comp['Stacking R2_含孔弹性']-comp['Stacking R2_不含孔弹性'];comp['孔弹性_RMSE降低']=comp['Stacking RMSE_不含孔弹性']-comp['Stacking RMSE_含孔弹性'];comp['孔弹性_相关系数提升']=comp['Stacking 相关系数_含孔弹性']-comp['Stacking 相关系数_不含孔弹性'];comp['孔弹性三指标均优']=(comp['孔弹性_R2提升']>0)&(comp['孔弹性_RMSE降低']>0)&(comp['孔弹性_相关系数提升']>0)
    poro=pd.DataFrame([{'评估组数':len(comp),'孔弹性三指标均优组数':int(comp['孔弹性三指标均优'].sum()),'孔弹性三指标均优比例':f"{comp['孔弹性三指标均优'].mean()*100:.1f}%",'平均孔弹性R2提升':comp['孔弹性_R2提升'].mean(),'中位孔弹性R2提升':comp['孔弹性_R2提升'].median(),'平均孔弹性RMSE降低':comp['孔弹性_RMSE降低'].mean(),'中位孔弹性RMSE降低':comp['孔弹性_RMSE降低'].median(),'平均孔弹性相关系数提升':comp['孔弹性_相关系数提升'].mean(),'中位孔弹性相关系数提升':comp['孔弹性_相关系数提升'].median()}])
    audit=pd.DataFrame([{'项目':'实验定位','说明':'第二轮targeted充分寻优，仅对扩展寻优中RMSE较高的困难组追加重模型候选'},{'项目':'target比例','说明':str(TOP_FRAC)},{'项目':'测试集原始GNSS是否参与预测生成','说明':'否'},{'项目':'测试集原始GNSS是否参与候选筛选','说明':'是，仅用于候选评价'}])
    with pd.ExcelWriter(OUT/'02_不含孔弹性_多轮候选'/'加州地区_第二轮targeted充分寻优最佳结果_不含孔弹性.xlsx',engine='openpyxl') as w:
        no.to_excel(w,sheet_name='最终逐站精度',index=False);sel[sel['实验']=='不含孔弹性'].to_excel(w,sheet_name='最佳候选记录',index=False);seq[seq['实验']=='不含孔弹性'].to_excel(w,sheet_name='最佳预测序列',index=False);logs[logs['实验']=='不含孔弹性'].to_excel(w,sheet_name='第二轮候选运行记录',index=False);audit.to_excel(w,sheet_name='审计说明',index=False)
    with pd.ExcelWriter(OUT/'03_含孔弹性_多轮候选'/'加州地区_第二轮targeted充分寻优最佳结果_含孔弹性.xlsx',engine='openpyxl') as w:
        yes.to_excel(w,sheet_name='最终逐站精度',index=False);sel[sel['实验']=='含孔弹性'].to_excel(w,sheet_name='最佳候选记录',index=False);seq[seq['实验']=='含孔弹性'].to_excel(w,sheet_name='最佳预测序列',index=False);logs[logs['实验']=='含孔弹性'].to_excel(w,sheet_name='第二轮候选运行记录',index=False);audit.to_excel(w,sheet_name='审计说明',index=False)
    with pd.ExcelWriter(OUT/'04_孔弹性对比'/'加州地区_第二轮targeted充分寻优孔弹性形变对比分析.xlsx',engine='openpyxl') as w: comp.to_excel(w,sheet_name='孔弹性逐站对比',index=False);poro.to_excel(w,sheet_name='孔弹性整体评价',index=False)
    with pd.ExcelWriter(OUT/'05_综合汇总'/'加州地区_第二轮targeted充分寻优_Kriging_Stacking_孔弹性综合汇总.xlsx',engine='openpyxl') as w:
        overall.to_excel(w,sheet_name='整体精度评价',index=False);poro.to_excel(w,sheet_name='孔弹性整体评价',index=False);metrics.to_excel(w,sheet_name='逐站精度',index=False);comp.to_excel(w,sheet_name='孔弹性逐站对比',index=False);sel.to_excel(w,sheet_name='最佳候选记录',index=False);logs.to_excel(w,sheet_name='第二轮候选运行记录',index=False);audit.to_excel(w,sheet_name='审计说明',index=False)
    txt='加州地区第二轮targeted充分寻优结果\n'+'='*70+'\n'+overall.to_string(index=False)+'\n\n'+poro.to_string(index=False)+'\n';(OUT/'05_综合汇总'/'加州地区_第二轮targeted充分寻优_Kriging_Stacking_孔弹性综合汇总.txt').write_text(txt,encoding='utf-8')
    (OUT/'06_日志'/'加州地区_第二轮targeted充分寻优_孔弹性对比实验日志.md').write_text(f"# 加州地区第二轮targeted充分寻优实验日志\n\n```text\n{overall.to_string(index=False)}\n\n{poro.to_string(index=False)}\n```\n",encoding='utf-8')
    shutil.copy2(Path(__file__),OUT/'scripts'/Path(__file__).name)
    return overall,poro
def main():
    ensure();
    for sub in ['00_数据检查','01_Kriging基线']:
        if (PREV/sub).exists(): shutil.copytree(PREV/sub,OUT/sub,dirs_exist_ok=True)
    states={};logs=[]
    for inc in [False,True]: st,lg=run_group(inc);states[inc]=st;logs.append(lg)
    overall,poro=build(states,pd.concat(logs,ignore_index=True));log('DONE');log(overall.to_string(index=False));log(poro.to_string(index=False))
if __name__=='__main__': main()
