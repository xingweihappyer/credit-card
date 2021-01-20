




import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
logging.info('hh')

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from hyperopt import fmin, tpe, hp

from lightgbm import LGBMClassifier
import lightgbm
import lightgbm as lgb
from  toad.metrics import KS,KS_bucket,AUC,PSI
from toad.selection import select

import perform as pf #载入自定义函数
from lgbm_tuner import *



"""
全部反欺诈通过进件打分
每个bin的占比
每个bin的风险
20*20
两个auc效果
"""
"""
select distinct apply_time,cust_id,mid,biz_id, loan_time,overdue_days,mb_cust_id,rank 
,extract(hour from apply_time)  as apply_hour  
from tmp.xw_application 
where cnt =1 
and date(apply_time)>=date('2020-11-01') and  date(apply_time)<=date('2020-12-20')
and (anti_result= 'PASSED' or anti_result2= 'PASSED')

"""

# v1 score 等频切分

def v1_tmp(v1_score):
    v1_bin=1
    if v1_score <= 467:
        v1_bin = 20
    elif v1_score > 467.0 and v1_score <= 474.0:
        v1_bin = 19
    elif v1_score > 474.0 and v1_score <= 479.0:
        v1_bin = 18
    elif v1_score > 479.0 and v1_score <= 483.0:
        v1_bin = 17
    elif v1_score > 483.0 and v1_score <= 487.0:
        v1_bin = 16
    elif v1_score > 487.0 and v1_score <= 491.0:
        v1_bin = 15
    elif v1_score > 491.0 and v1_score <= 494.0:
        v1_bin = 14
    elif v1_score > 494.0 and v1_score <= 498.0:
        v1_bin = 13
    elif v1_score > 498.0 and v1_score <= 502.0:
        v1_bin = 12
    elif v1_score > 502.0 and v1_score <= 505.0:
        v1_bin = 11
    elif v1_score > 505.0 and v1_score <= 509.0:
        v1_bin = 10
    elif v1_score > 509.0 and v1_score <= 513.0:
        v1_bin = 9
    elif v1_score > 513.0 and v1_score <= 518.0:
        v1_bin = 8
    elif v1_score > 518.0 and v1_score <= 523.0:
        v1_bin = 7
    elif v1_score > 523.0 and v1_score <= 530.0:
        v1_bin = 6
    elif v1_score > 530.0 and v1_score <= 537.0:
        v1_bin = 5
    elif v1_score > 537.0 and v1_score <= 547.0:
        v1_bin = 4
    elif v1_score > 547.0 and v1_score <= 560.0:
        v1_bin = 3
    elif v1_score > 560.0 and v1_score <= 580.0:
        v1_bin = 2
    elif v1_score > 580.0:
        v1_bin = 1
    return v1_bin

df_v1 =  pd.read_csv('/home/bb5/xw/mxg_scores/20210112_pg_v1_scores.csv',encoding='gbk')
df_v1['bin'],_ = pd.qcut(df_v1['v1_score'],20, retbins=True)
df_v1['v1_bin'] = df_v1['v1_score'].map(v1_tmp)
# df_v1.groupby(['bin'])['biz_id'].count()



# v3 数据读取
df_v3 = pd.read_csv('/home/bb5/xw/mxg_scores/20210112_pg_apply.csv',encoding='gbk')
df_v3_oot = df_v3[df_v3['overdue_days']>=-9]
df_v3_oot['y'] = df_v3_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)



factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
logger.info('factor:{} offset:{}'.format(factor,offset))



path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))

X =df_v3
df_v3['preds'] = lgbm_model.predict(X.loc[:, chosen_feature])
df_v3['score'] =  (np.log((1 - df_v3['preds'])/df_v3['preds']) * factor + offset).round(0)

def v3_bin(v3_score):
    if v3_score <= 471:
        bin = 20
    elif v3_score > 471 and v3_score <= 482:
        bin = 19
    elif v3_score > 482 and v3_score <= 490:
        bin = 18
    elif v3_score > 490 and v3_score <= 498:
        bin = 17
    elif v3_score > 498 and v3_score <= 504:
        bin = 16
    elif v3_score > 504 and v3_score <= 510:
        bin = 15
    elif v3_score > 510 and v3_score <= 517:
        bin = 14
    elif v3_score > 517 and v3_score <= 523:
        bin = 13
    elif v3_score > 523 and v3_score <= 529:
        bin = 12
    elif v3_score > 529 and v3_score <= 536:
        bin = 11
    elif v3_score > 536 and v3_score <= 542:
        bin = 10
    elif v3_score > 542 and v3_score <= 550:
        bin = 9
    elif v3_score > 550 and v3_score <= 557:
        bin = 8
    elif v3_score > 557 and v3_score <= 565:
        bin = 7
    elif v3_score > 565 and v3_score <= 574:
        bin = 6
    elif v3_score > 574 and v3_score <= 584:
        bin = 5
    elif v3_score > 584 and v3_score <= 596:
        bin = 4
    elif v3_score > 596 and v3_score <= 611:
        bin = 3
    elif v3_score > 611 and v3_score <= 632:
        bin = 2
    elif v3_score > 632:
        bin = 1
    return bin



df_v3['v3_bin'] = df_v3['score'].map(v3_bin)

df_v3.groupby(['v3_bin'])['score'].count()

X,Y = df_v3_oot,df_v3_oot['y']
v3_preds = lgbm_model.predict(X.loc[:, chosen_feature])
df_v3_oot['v3_preds'] = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(v3_preds,Y)
auc = AUC(v3_preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))

df_v3_oot['score'] = (np.log((1 - df_v3['preds'])/df_v3['preds']) * factor + offset).round(0)
v3_ks_bucket = KS_bucket(df_v3_oot['score'],df_v3_oot['y'],bucket=10,method='quantile')
v3_ks_bucket[['min', 'max', 'bads', 'goods', 'total', 'bad_rate']].to_csv('/home/bb5/xw/mxg_scores/v3_0112.csv')
# 读取model文件，KS值和AUC值分别为[0.3276918639284987, 0.7198946953209765]

# v1 效果

df_v1_f = pd.read_csv('/home/bb5/xw/mxg_scores/20210112_pg_v1_f.csv',encoding='gbk')
df_v1_loan = pd.merge(df_v1_f,df_v3_oot,left_on='biz_ids',right_on='biz_id')
X,Y = df_v1_loan.loc[:,'f0':'f31'],df_v1_loan['y']
input_model_path='/home/bb5/xw/mxg_scores/model_v1/lgb.model'
lgbm_model = lgb.Booster(model_file=input_model_path)
v1_preds = lgbm_model.predict(X)
df_v1_loan['v1_preds'] = lgbm_model.predict(X)
ks = KS(v1_preds,Y)
auc = AUC(v1_preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# 读取model文件，KS值和AUC值分别为[0.19721778440566945, 0.6296841006625415]



df_v1_loan['score'] = (np.log((1 - df_v1_loan['v1_preds'])/df_v1_loan['v1_preds']) * factor + offset).round(0)
v1_ks_bucket = KS_bucket(df_v1_loan['score'],df_v1_loan['y'],bucket=10,method='quantile')
v1_ks_bucket[['min', 'max', 'bads', 'goods', 'total', 'bad_rate']].to_csv('/home/bb5/xw/mxg_scores/v1_0112.csv')

# 计算corr

df_v1_v3 = pd.merge(df_v1_loan,df_v3_oot,left_on='biz_ids',right_on='biz_id')

pccs = np.corrcoef(list(v1_preds), list(v3_preds))
from scipy.stats import pearsonr
pccs = pearsonr(df_v1_v3['v1_preds'], df_v1_v3['v3_preds_y'])

# v1 v3 关联

df = pd.merge(df_v3,df_v1,left_on='biz_id',right_on='biz_id')
df = df.drop_duplicates()

# 进件二维矩阵
apply_bin = df.groupby(['v3_bin','v1_bin'])['biz_id'].count()
apply_bin =apply_bin.reset_index()
apply_bin.pivot(index='v3_bin',columns='v1_bin',values='biz_id').to_csv('/home/bb5/xw/mxg_scores/strategy/apply_v1v3.csv')


# 贷后表现 T7
df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)
df_oot = df_oot.groupby(['v3_bin','v1_bin'])['y'].mean()
df_oot =df_oot.reset_index()
df_oot.pivot(index='v3_bin',columns='v1_bin',values='y').to_csv('/home/bb5/xw/mxg_scores/strategy/T7_v1v3.csv')

# 贷后表现 T1
df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=1 else 0)
df_oot = df_oot.groupby(['v3_bin','v1_bin'])['y'].mean()
df_oot =df_oot.reset_index()
df_oot.pivot(index='v3_bin',columns='v1_bin',values='y').to_csv('/home/bb5/xw/mxg_scores/strategy/T1_v1v3.csv')

# 放款件
df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)
df_loan = df_oot.groupby(['v3_bin','v1_bin'])['y'].count()
df_loan =df_loan.reset_index()
df_loan.pivot(index='v3_bin',columns='v1_bin',values='y').to_csv('/home/bb5/xw/mxg_scores/strategy/loan_v1v3.csv')

# 逾期件
df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)
df_loan = df_oot.groupby(['v3_bin','v1_bin'])['y'].sum()
df_loan =df_loan.reset_index()
df_loan.pivot(index='v3_bin',columns='v1_bin',values='y').to_csv('/home/bb5/xw/mxg_scores/strategy/due_v1v3.csv')


# lift
df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)
df_oot.groupby(['v1_bin'])['y'].mean()
df_oot.groupby(['v3_bin'])['y'].mean()

df_oot.groupby(['v1_bin'])['y'].count()
df_oot.groupby(['v3_bin'])['y'].count()

df_oot = df[df['overdue_days']>=-9]
df_oot['y'] = df_oot['overdue_days'].map(lambda x: 1 if x>=1 else 0)
df_oot.groupby(['v1_bin'])['y'].mean()
df_oot.groupby(['v3_bin'])['y'].mean()
df_oot.groupby(['v3_bin'])['y'].count()
df_oot.groupby(['v3_bin'])['y'].sum()

# swap



#  是否使用催收

# v3 数据读取
df_v3 = pd.read_csv('/home/bb5/xw/mxg_scores/20210112_pg_apply.csv',encoding='gbk')
df_v3_oot = df_v3[df_v3['overdue_days']>=1]
df_v3_oot['y'] = df_v3_oot['overdue_days'].map(lambda x: 1 if x>=7 else 0)

factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
logger.info('factor:{} offset:{}'.format(factor,offset))


path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))

X,Y = df_v3_oot,df_v3_oot['y']
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# 读取model文件，KS值和AUC值分别为[0.18719921198757752, 0.6244809881666838]
KS_bucket(preds,Y,bucket=10,method='quantile').to_csv(path+'/催收是否适用.csv')


# v1 效果

df_v1_f = pd.read_csv('/home/bb5/xw/mxg_scores/20210112_pg_v1_f.csv',encoding='gbk')
df_v1_loan = pd.merge(df_v1_f,df_v3_oot,left_on='biz_ids',right_on='biz_id')
df_v1_loan = df_v1_loan[df_v1_loan['overdue_days']>=1]

X,Y = df_v1_loan.loc[:,'f0':'f31'],df_v1_loan['y']
input_model_path='/home/bb5/xw/mxg_scores/model_v1/lgb.model'
lgbm_model = lgb.Booster(model_file=input_model_path)
v1_preds = lgbm_model.predict(X)
df_v1_loan['v1_preds'] = lgbm_model.predict(X)
ks = KS(v1_preds,Y)
auc = AUC(v1_preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# 读取model文件，KS值和AUC值分别为[0.11429507277130459, 0.570581572611093]
KS_bucket(preds,Y,bucket=10,method='quantile').to_csv(path+'/v1催收是否适用.csv')
