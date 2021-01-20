

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
logging.info('hh')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold  # 交叉验证
from hyperopt import fmin, tpe, hp

from lightgbm import LGBMClassifier
import lightgbm
import lightgbm as lgb
import toad

import perform as pf #载入自定义函数
from lgbm_tuner import *


"""
检查数值是否异常
train+test是2w
oot是0.5W
"""

sample = pd.read_csv('/home/bb5/xw/mxg20201113.csv',encoding='gbk')
sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)
# sample.describe().to_csv('/home/bb5/xw/sample_check.csv')
# print(sample.dtypes)

# 拆分数据
sample01=sample[sample['apply_time']<='2020-09-24']
y=sample01['y']
sample01.drop(columns = ['apply_time','biz_id','cust_id','overdue_days','y'],inplace=True)
x = sample01
# print(x.dtypes)
print('训练样本逾期率',y.sum()/y.count())



oot=sample[sample['apply_time']>='2020-09-25']
oot_y = oot['y']
oot.drop(columns = ['apply_time','biz_id','cust_id','overdue_days','y'],inplace=True)
oot_x = oot
# print(oot_x.dtypes)
print('oot样本逾期率',oot_y.sum()/oot_y.count())


# 划分train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# """
# 3.定义贝叶斯调参的目标函数
# """
# kf = KFold(n_splits=5, random_state=7, shuffle=True)
# def ksscore(args):
#     print(args)
#     ks = []
#     chosen_feature = feat_imp.index[:int(args['feature_num'])]  # 选取feature_importance排在前feature_num的变量
#     X_use = X.loc[:, chosen_feature]
#
#     for train_index, test_index in kf.split(X):
#         x_train_kf, x_test_kf = X_use.iloc[train_index], X_use.iloc[test_index]
#         y_train_kf, y_test_kf = Y.iloc[train_index], Y.iloc[test_index]
#
#         lgbm_model = LGBMClassifier(
#             learning_rate=args['learning_rate'],
#             n_estimators=300,
#             max_depth=int(args['max_depth']),
#             # max_depth=2,
#             min_child_weight=70,
#             min_split_gain=args['min_split_gain'],
#             # min_child_samples=args['min_child_samples'],
#             subsample=1,
#             colsample_bytree=1,
#             objective='binary',
#             reg_alpha=args['reg_alpha'],
#             reg_lambda=args['reg_lambda'],
#             seed=7)
#
#         lgbm_param_temp = lgbm_model.get_params()
#
#         lgbm_param_temp.pop('silent')
#         lgbm_param_temp.pop('n_estimators')
#
#         lgbm_train = lgb.Dataset(x_train_kf, y_train_kf)
#         cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=300, nfold=5, metrics='auc',
#                           early_stopping_rounds=100)
#         best_n_estimators = len(cvresult['auc-mean'])
#         lgbm_model.set_params(n_estimators=best_n_estimators)
#
#         lgbm_model.fit(x_train_kf, y_train_kf)
#         y_pred_test = lgbm_model.predict(x_test_kf)
#         ks += [max(pf.cal_ks(-y_pred_test, y_test_kf)[0])]
#
#     score = np.mean(np.array(ks)) - 1.96 * np.std(np.array(ks)) / np.sqrt(len(ks))
#     print(score)
#     return -score
#
# '''feature_num： 这个表示变量个数，70个变量起到100个， 5个递增，可以根据要求改，比如30等'''
# para_space_mlp={'feature_num':hp.quniform('feature_num',60,100,5),
#                 'learning_rate':hp.quniform('learning_rate',0.01,0.1,0.001),
#                 'max_depth':hp.quniform('max_depth',2,3,1),
#                 #'min_child_weight':hp.quniform('min_child_weight',1,20,1),
#                 'min_split_gain':hp.quniform('min_split_gain',1,6,0.1),
#                 #'min_child_samples':hp.quniform('min_child_samples',20,50,2),
#                 #'subsample':hp.quniform('subsample',0.6,1,0.1),
#                 #'colsample_bytree':hp.quniform('colsample_bytree',0.6,1,0.1),
#                 'reg_alpha':hp.quniform('reg_alpha',0,2,0.01),
#                 'reg_lambda': hp.quniform('reg_lambda',7,20,0.1),
#                 }
#
# # starttime = datetime.datetime.now()
#
# #max_evals为调参的步数，根据数据量大小适当调整
# best = fmin(ksscore, para_space_mlp, algo=tpe.suggest, max_evals=100,rstate=np.random.RandomState(7))
#

parameters={'num_leaves':[i for i in range(10,50,1)],
            'max_depth':[2,3,4,5,6],
            'learning_rate':[0.001,0.003,0.005]+[i/100 for i in range(1,11)],
            'n_estimators':range(100,500,2),
            'min_split_gain':[i/100 for i in range(0,10,1)],
            'min_child_weight':[0.001,0.003,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
            'min_child_samples':[i for i in range(10,50,1)],
            'subsample':[i/10 for i in range(6,11)],
            'colsample_bytree':[i/10 for i in range(6,11)],
            'reg_alpha':[i/100 for i in range(0,200,1)],
            'reg_lambda':[i/10 for i in range(70,200,1)]
            }


feat_imp = init_feature(x_train, y_train)
valid_feature = list(feat_imp[feat_imp>0].index)
best_para = grid_search(parameters,x_train[valid_feature], x_test[valid_feature], y_train, y_test)
