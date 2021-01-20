

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

sample = sample[sample['overdue_days']>=1]
sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)
# sample.describe().to_csv('/home/bb5/xw/sample_check.csv')
# print(sample.dtypes)

sample.groupby('apply_time')['y'].count().to_csv('day_cnt.csv')
sample.to_csv('/home/bb5/xw/mxg_scores/test.csv')
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
valid_f = valid_f[valid_f<0.2].index


# 获取变量重要性
# feat_imp = init_feature(x_train.loc[:,valid_f], y_train)

"""
3.定义贝叶斯调参的目标函数
"""
X,Y=x_train,y_train
kf = KFold(n_splits=5, random_state=7, shuffle=True)

def ksscore(args):
    X_use = X.loc[:, list(valid_f)]
    lgbm_model = LGBMClassifier(
        learning_rate=args['learning_rate'],
        n_estimators=400,
        max_depth=int(args['max_depth']),
        # max_depth=2,
        min_child_weight=70,
        min_split_gain=args['min_split_gain'],
        # min_child_samples=args['min_child_samples'],
        subsample=1,
        colsample_bytree=1,
        objective='binary',
        reg_alpha=args['reg_alpha'],
        reg_lambda=args['reg_lambda'],
        seed=7)

    lgbm_param_temp = lgbm_model.get_params()

    lgbm_param_temp.pop('silent')
    lgbm_param_temp.pop('n_estimators')

    lgbm_train = lgb.Dataset(X_use, Y)
    cvresult = lgb.cv(lgbm_param_temp, lgbm_train, num_boost_round=50, nfold=5, metrics='auc',
                      early_stopping_rounds=50)
    best_score = max(cvresult['auc-mean'])
    loss = 1 - best_score

    return loss

'''feature_num： 这个表示变量个数，70个变量起到100个， 5个递增，可以根据要求改，比如30等'''
para_space_mlp={
                'learning_rate':hp.quniform('learning_rate',0.01,0.1,0.001),
                'max_depth':hp.quniform('max_depth',2,6,1),
                'min_split_gain':hp.quniform('min_split_gain',0,0.1,0.01),
                'min_child_samples':hp.quniform('min_child_samples',20,50,2),
                'subsample':hp.quniform('subsample',0.7,1,0.1),
                'colsample_bytree':hp.quniform('colsample_bytree',0.7,1,0.1),
                'reg_alpha':hp.quniform('reg_alpha',0,10,0.1),
                'reg_lambda': hp.quniform('reg_lambda',0,20,0.1),
                }


#max_evals为调参的步数，根据数据量大小适当调整
best = fmin(ksscore, para_space_mlp, algo=tpe.suggest, max_evals=10,rstate=np.random.RandomState(7))


#贝叶斯调参，确定最优参数 【等上面100%跑完跑这个】
lgbm_model = LGBMClassifier(
                    learning_rate =best['learning_rate'],
                    n_estimators=400,
                    max_depth=int(best['max_depth']),
                    min_split_gain=best['min_split_gain'],
                    min_child_samples=int(best['min_child_samples']),
                    subsample=best['subsample'],
                    colsample_bytree=best['colsample_bytree'],
                    objective= 'binary',
                    reg_alpha=best['reg_alpha'],
                    reg_lambda=best['reg_lambda'],
                    seed=7,random_state=7)

chosen_final_feature= list(valid_f)

lgbm_param = lgbm_model.get_params()

lgbm_param.pop('silent')
lgbm_param.pop('n_estimators')

lgbm_train = lgb.Dataset(X.loc[:, chosen_final_feature], Y)
cvresult = lgb.cv(lgbm_param, lgbm_train, num_boost_round=100, nfold=5, metrics='auc', early_stopping_rounds=50)
best_n_estimators = len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)

print('lightgbm模型调参完成！最终参数：')
print(lgbm_model.get_params())





X_use = X.loc[:, chosen_final_feature]
lgbm_model.fit(X_use, Y, eval_metric=['auc','binary_logloss'])

# train
X,Y = x_train,y_train
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('train，KS值和AUC值分别为{0}'.format([ks, auc]))


# test
X,Y = x_test,y_test
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('test，KS值和AUC值分别为{0}'.format([ks, auc]))


# oot1
X,Y = oot1_x,oot1_y
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot1，KS值和AUC值分别为{0}'.format([ks, auc]))


# oot2
X,Y = oot2_x,oot2_y
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot2，KS值和AUC值分别为{0}'.format([ks, auc]))




# 保存重要性
