


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
检查数值是否异常
train+test是2w
oot是0.5W
"""

# 读取数据
sample01 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part1.csv',encoding='gbk')
sample02 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part2.csv',encoding='gbk')
sample =pd.merge(sample01,sample02,how='inner',left_on='biz_id',right_on='biz_ids')

del sample01,sample02

sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)
# sample.describe().to_csv('/home/bb5/xw/sample_check.csv')
# print(sample.dtypes)

sample.groupby('apply_time')['y'].count().to_csv('day_cnt.csv')

# train+test
sample01=sample[sample['apply_time']<='2020-10-09']
sample01.drop(columns = ['apply_time','biz_id','cust_id','overdue_days','biz_ids'],inplace=True)
sample01.drop(columns = ['base_info_update_days'],inplace=True)



# 变量初步刷选
sample02 = select(sample01,target='y', empty=0.9, iv=0.02, corr=0.9)


y=sample02['y']
sample02.drop(columns = ['y'],inplace=True)
x = sample02
print('训练样本数',y.count())
print('训练样本逾期率',y.sum()/y.count())




# oot1
oot1=sample[(sample['apply_time']>='2020-10-10') & (sample['apply_time']<='2020-10-20')]
oot1_y = oot1['y']
oot1.drop(columns = ['apply_time','biz_id','cust_id','overdue_days','y','biz_ids'],inplace=True)
oot1_x = oot1
print('oot1样本数',oot1_y.count())
print('oot1样本逾期率',oot1_y.sum()/oot1_y.count())


# oot2
oot2=sample[sample['apply_time']>='2020-10-21']
oot2_y = oot2['y']
oot2.drop(columns = ['apply_time','biz_id','cust_id','overdue_days','y','biz_ids'],inplace=True)
oot2_x = oot2
print('oot2样本数',oot2_y.count())
print('oot2样本逾期率',oot2_y.sum()/oot2_y.count())





# 划分train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# psi稳定性刷选
valid_f = PSI(x_train,x_test)
valid_f = valid_f[valid_f<0.25].index


# 获取变量重要性
feat_imp = init_feature(x_train.loc[:,valid_f], y_train)

# 'min_split_gain': [i / 100 for i in range(0, 10, 1)],
# 'min_child_samples': [i for i in range(10, 50, 1)],
# 'subsample': [i / 10 for i in range(6, 11)],
# 'colsample_bytree': [i / 10 for i in range(6, 11)],
# 'reg_alpha': [i / 100 for i in range(0, 200, 1)],
# 'reg_lambda': [i / 10 for i in range(70, 200, 1)]
# 'learning_rate': [0.001, 0.003, 0.005] + [i / 100 for i in range(1, 11)],
# 'n_estimators': range(100, 500, 2),

from tqdm import tqdm

# 第一步，确定最优参数  num_leaves  max_depth
parameters_01={'num_leaves':[i for i in range(10,50,2)],
            'max_depth':[2,3,4,5,6]}

lgbm_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=4,
    min_split_gain=0.01,
    min_child_samples=20,
    num_leaves=50,
    min_child_weight=0.01,
    colsample_bytree=1,
    objective='binary',
    reg_alpha=0,
    reg_lambda=0,
    random_state=7)


gsearch = GridSearchCV(lgbm_model, param_grid=parameters_01, scoring='roc_auc', cv=3,verbose=10)
gsearch.fit(x_train, y_train,verbose=False)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

best_num_leaves = gsearch.best_params_['num_leaves']
best_max_depth = gsearch.best_params_['max_depth']


# 第二步，确定最优参数  reg_alpha，reg_lambda

parameters_02={'reg_alpha': [i / 100 for i in range(10, 200, 10)],
            'reg_lambda': [i / 10 for i in range(10, 100, 5)]}

lgbm_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=best_max_depth,
    min_split_gain=0.01,
    min_child_samples=20,
    subsample=1,
    num_leaves=best_num_leaves,
    min_child_weight=0.01,
    colsample_bytree=1,
    objective='binary',
    reg_alpha=0,
    reg_lambda=0,
    random_state=7)


gsearch = GridSearchCV(lgbm_model, param_grid=parameters_02, scoring='roc_auc', cv=3,verbose=10)
gsearch.fit(x_train, y_train,verbose=False)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

best_reg_alpha = gsearch.best_params_['reg_alpha']
best_reg_lambda = gsearch.best_params_['reg_lambda']



# 第3步，确定最优参数  reg_alpha，reg_lambda

parameters_03={'subsample': [i / 10 for i in range(7,11)],
            'colsample_bytree': [i / 10 for i in range(7,11)]}

lgbm_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=best_max_depth,
    min_split_gain=0.01,
    min_child_samples=20,
    subsample=1,
    num_leaves=best_num_leaves,
    min_child_weight=0.01,
    colsample_bytree=1,
    objective='binary',
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
    random_state=7)


gsearch = GridSearchCV(lgbm_model, param_grid=parameters_03, scoring='roc_auc', cv=3,verbose=10)
gsearch.fit(x_train, y_train,verbose=False)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))

best_colsample_bytree=gsearch.best_params_['colsample_bytree']
best_subsample=gsearch.best_params_['subsample']



# 第4步，确定最优参数  reg_alpha，reg_lambda

parameters_04={'min_split_gain': [i / 1000 for i in range(0,5)],
               'min_child_samples': [i *10 for i in range(1,6)],
               'min_child_weight':[i / 1000 for i in range(1,5)]
               }

lgbm_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=best_max_depth,
    min_split_gain=0.01,
    min_child_samples=20,
    subsample=best_subsample,
    num_leaves=best_num_leaves,
    min_child_weight=0.01,
    colsample_bytree=best_colsample_bytree,
    objective='binary',
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
    random_state=7)


gsearch = GridSearchCV(lgbm_model, param_grid=parameters_04, scoring='roc_auc', cv=3,verbose=10)
gsearch.fit(x_train, y_train,verbose=False)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))



best_min_child_samples = gsearch.best_params_['min_child_samples']
best_min_child_weight = gsearch.best_params_['min_child_weight']
best_min_split_gain = gsearch.best_params_['min_split_gain']


# 第5步，确定最优参数  reg_alpha，reg_lambda

parameters_05={'learning_rate':[i / 100 for i in range(1,20,2)],
               'n_estimators': [i  for i in range(100, 300,4)],}

lgbm_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=best_max_depth,
    min_split_gain=best_min_split_gain,
    min_child_samples=best_min_child_samples,
    subsample=best_subsample,
    num_leaves=best_num_leaves,
    min_child_weight=best_min_child_weight,
    colsample_bytree=best_colsample_bytree,
    objective='binary',
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
    random_state=7)


gsearch = GridSearchCV(lgbm_model, param_grid=parameters_05, scoring='roc_auc', cv=3,verbose=10)
gsearch.fit(x_train, y_train,verbose=False)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))



best_learning_rate = gsearch.best_params_['learning_rate']
best_n_estimators = gsearch.best_params_['n_estimators']


lgbm_model = LGBMClassifier(
    learning_rate=best_learning_rate,
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_split_gain=best_min_split_gain,
    min_child_samples=best_min_child_samples,
    subsample=best_subsample,
    num_leaves=best_num_leaves,
    min_child_weight=best_min_child_weight,
    colsample_bytree=best_colsample_bytree,
    objective='binary',
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
    random_state=7)




# eval_metric='auc'
lgbm_model.fit(x_train,y_train,)


# train
X,Y = x_train,y_train
preds = lgbm_model.predict_proba(X.loc[:, ])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('train，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.6101737528350761, 0.8649091690440008]
# [0.6525369650376067, 0.8862807934521714]
# train，KS值和AUC值分别为[0.31283406586210283, 0.7126621830832252]


# test
X,Y = x_test,y_test
preds = lgbm_model.predict_proba(X.loc[:, ])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('test，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.5977482436176529, 0.8566478417014936]
# [0.6898439660019113, 0.9150092752650243]
# [0.31988796801011776, 0.7128375467400838]

# oot1
X,Y = oot1_x,oot1_y
preds = lgbm_model.predict_proba(X.loc[:, x_train.columns])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot1，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.7463033314023115, 0.935975470791987]
# [0.6715034073245776, 0.8911824402804956]
# [0.33712799308429264, 0.7211240343142982]

# oot2
X,Y = oot2_x,oot2_y
preds = lgbm_model.predict_proba(X.loc[:, x_train.columns])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot2，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.6034645021088599, 0.8592923612387786]
#  [0.4371277291242013, 0.7895184718592859]

