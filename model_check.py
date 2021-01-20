


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


# 读取数据
sample= pd.read_csv('/home/bb5/xw/mxg_scores/20201123_pg_part.csv',encoding='gbk')
sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)


# 导入模型变量
# 读取保存好的 model 和feature names
path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))



# oot1
oot1=sample[(sample['apply_time']>='2020-10-10') & (sample['apply_time']<='2020-10-20')]
oot1_y = oot1['y']
oot1_x = oot1
print('oot1样本数',oot1_y.count())
print('oot1样本逾期率',oot1_y.sum()/oot1_y.count())


# oot2
oot2=sample[(sample['apply_time']>='2020-10-21') & (sample['apply_time']<='2020-11-01') ]
oot2_y = oot2['y']
oot2_x = oot2
print('oot2样本数',oot2_y.count())
print('oot2样本逾期率',oot2_y.sum()/oot2_y.count())


# oot3
oot3=sample[(sample['apply_time']>='2020-11-02') & (sample['apply_time']<='2020-11-07') ]
oot3_y = oot3['y']
oot3_x = oot3
print('oot3样本数',oot3_y.count())
print('oot3样本逾期率',oot3_y.sum()/oot3_y.count())



"""
模型效果重新验证 
"""
X,Y = oot1_x,oot1_y
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
#[0.3301062511292491, 0.7198242221251013]

X,Y = oot2_x,oot2_y
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.4384826660424955, 0.7895857852596792]

X,Y = oot3_x,oot3_y
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.33491709884462445, 0.7203960446872948]




"""
发现 oot1 有点偏差，查看变量值是否发生变化
"""

# 读取数据
sample01 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part1.csv',encoding='gbk')
sample02 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part2.csv',encoding='gbk')
sample_check =pd.merge(sample01,sample02,how='inner',left_on='biz_id',right_on='biz_ids')
sample_check['y']=sample_check['overdue_days'].map(lambda x: 1 if x>=7 else 0)
feature_tmp = chosen_feature+['biz_id','y','overdue_days','apply_time']
sample_check = sample_check.loc[:, feature_tmp]
oot1_check=sample_check[(sample_check['apply_time']>='2020-10-10') & (sample_check['apply_time']<='2020-10-20')]



# 关联两张表 biz_id

df_check  =pd.merge(oot1_check,oot1,how='inner',left_on='biz_id',right_on='biz_id')

df_check.fillna(-999,inplace=True)
feature = chosen_feature+['y']

for f in  feature:
    f1 = f + '_x'
    f2 = f + '_y'
    check = sum(df_check[f1] !=  df_check[f2])
    rate = sum(df_check[f1] !=  df_check[f2])/len(df_check)
    logger.info('{} , {} , {}'.format(f,check,rate))


