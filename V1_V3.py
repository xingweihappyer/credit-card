


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
全部安装进件打分
每个bin的占比
每个bin的风险
20*20
两个auc效果
"""

# 整个进件的的v1 v3的bin
df = pd.read_csv('/home/bb5/xw/mxg_scores/20210106_pg_pred.csv',encoding='gbk')
df_v1 = pd.read_csv('/home/bb5/xw/mxg_scores/20210106_pg_v1_bin.csv',encoding='gbk')

factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
logger.info('factor:{} offset:{}'.format(factor,offset))



path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))

X =df
df['preds'] = lgbm_model.predict(X.loc[:, chosen_feature])
df['score'] =  (np.log((1 - df['preds'])/df['preds']) * factor + offset).round(0)

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


df['v3_bin'] = df['score'].map(v3_bin)

df02 = pd.merge(df_v1,df,left_on='biz_id',right_on='biz_id',how='inner')

# 缺失率
logger.info(df02.isna().sum()/len(df02))

df_v1_v3 = df02.groupby(['v1_bin','v3_bin'])['biz_id'].count()
df_v1_v3 = df_v1_v3.reset_index()

v1_v3 = df_v1_v3.pivot(index ='v3_bin', columns ='v1_bin', values =['biz_id'])
v1_v3.to_csv(path+'/v1_v303.csv')





# 近期12.1-12.15的每个bin的逾期表现
# select  *  from tmp.v3_feature_all  where date(apply_time)>=date('2020-12-01') and date(apply_time)<=date('2020-12-15') and overdue_days>=-10
"""
select sum(case when overdue_days>=1 then 1 else 0 end )*1.0/count(*)
,sum(case when overdue_days>=7 then 1 else 0 end )*1.0/count(*)
from 
(select  *  from tmp.v3_feature_all  where date(apply_time)>=date('2020-12-01') and date(apply_time)<=date('2020-12-15') and overdue_days>=-10) a 
"""
df = pd.read_csv('/home/bb5/xw/mxg_scores/20210111_pg_loan.csv',encoding='gbk')


# 导入模型变量
# 读取保存好的 model 和feature names

path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))

X = df
df["preds"] = lgbm_model.predict(X.loc[:, chosen_feature])
df['score'] =  (np.log((1 - df['preds'])/df['preds']) * factor + offset).round(0)
df['bin'] = df['score'].map(v3_bin)
df['y_7'] = df['overdue_days'].map(lambda x: 1 if x>=7 else 0)
df['y_1'] = df['overdue_days'].map(lambda x: 1 if x>=1 else 0)

df.groupby(['bin'])['y_1'].mean()
df.groupby(['bin'])['y_7'].mean()








