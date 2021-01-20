

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
from  xgboost import XGBClassifier
import xgboost



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
valid_f = valid_f[valid_f<0.25].index



X,Y=x_train,y_train
kf = KFold(n_splits=3, random_state=7, shuffle=True)
def ksscore(args):
    X_use = X.loc[:, list(valid_f)]
    xgb_model = XGBClassifier(
        learning_rate=args['learning_rate'],
        n_estimators=300,
        max_depth=int(args['max_depth']),
        min_child_weight=args['min_child_weight'],
        min_split_loss=args['min_split_loss'],
        subsample=args['subsample'],
        colsample_bytree=1,
        objective='binary',
        reg_alpha=args['reg_alpha'],
        reg_lambda=args['reg_lambda'],
        seed=7)

    xgb_param_temp = xgb_model.get_params()

    xgb_param_temp.pop('silent')
    xgb_param_temp.pop('n_estimators')

    xgb_train = xgboost.DMatrix(X_use, Y)
    cvresult = lgb.cv(xgb_param_temp, xgb_train, num_boost_round=100, nfold=5, metrics='auc',
                      early_stopping_rounds=10)
    best_score = max(cvresult['auc-mean'])
    loss = 1 - best_score

    return loss


'''feature_num： 这个表示变量个数，70个变量起到100个， 5个递增，可以根据要求改，比如30等'''
para_space_mlp={'feature_num':hp.quniform('feature_num',20,40,1),
                'learning_rate':hp.quniform('learning_rate',0.01,0.1,0.001),
                'max_depth':hp.quniform('max_depth',2,4,1),
                'min_split_loss':hp.quniform('min_split_loss',0,0.1,0.01),
                'min_child_weight':hp.quniform('min_child_weight',20,50,2),
                'subsample':hp.quniform('subsample',0.7,1,0.1),
                'colsample_bytree':hp.quniform('colsample_bytree',0.7,1,0.1),
                'reg_alpha':hp.quniform('reg_alpha',0,10,0.1),
                'reg_lambda': hp.quniform('reg_lambda',0,20,0.1),
                }


#max_evals为调参的步数，根据数据量大小适当调整
best = fmin(ksscore, para_space_mlp, algo=tpe.suggest, max_evals=100,rstate=np.random.RandomState(7))





#贝叶斯调参，确定最优参数 【等上面100%跑完跑这个】
lgbm_model = LGBMClassifier(
                    learning_rate =best['learning_rate'],
                    n_estimators=300,
                    max_depth=int(best['max_depth']),
                    #max_depth = 2,
                    min_split_gain=best['min_split_gain'],
                    min_child_samples=int(best['min_child_samples']),
                    subsample=best['subsample'],
                    colsample_bytree=best['colsample_bytree'],
                    objective= 'binary',
                    reg_alpha=best['reg_alpha'],
                    reg_lambda=best['reg_lambda'],
                    seed=7,random_state=7)

chosen_final_feature=feat_imp.index[:int(best['feature_num'])]

lgbm_param = lgbm_model.get_params()

lgbm_param.pop('silent')
lgbm_param.pop('n_estimators')

lgbm_train = lgb.Dataset(X.loc[:, chosen_final_feature], Y)
cvresult = lgb.cv(lgbm_param, lgbm_train, num_boost_round=300, nfold=5, metrics='auc', early_stopping_rounds=100)
best_n_estimators = len(cvresult['auc-mean'])
lgbm_model.set_params(n_estimators=best_n_estimators)

print('lightgbm模型调参完成！最终参数：')
print(lgbm_model.get_params())

# 保存本地
import os
import json
os.remove("best_json.json")
best_c = json.dumps(best)
f2 = open('best_json.json', 'w')
f2.write(best_c)
f2.close()



# cv的 best_n_estimators 会变化
# {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.9, 'importance_type': 'split', 'learning_rate': 0.085, 'max_depth': 4, 'min_child_samples': 34, 'min_child_weight': 70, 'min_split_gain': 0.07, 'n_estimators': 143, 'n_jobs': -1, 'num_leaves': 31, 'objective': 'binary', 'random_state': None, 'reg_alpha': 5.5, 'reg_lambda': 13.700000000000001, 'silent': True, 'subsample': 0.9, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'seed': 7}
# {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05, 'max_depth': 6, 'min_child_samples': 26, 'min_child_weight': 70, 'min_split_gain': 0.03, 'n_estimators': 166, 'n_jobs': -1, 'num_leaves': 31, 'objective': 'binary', 'random_state': 7, 'reg_alpha': 3.7, 'reg_lambda': 16.1, 'silent': True, 'subsample': 0.7000000000000001, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'seed': 7}
# {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.8, 'importance_type': 'split', 'learning_rate': 0.07, 'max_depth': 4, 'min_child_samples': 36, 'min_child_weight': 0.001, 'min_split_gain': 0.04, 'n_estimators': 110, 'n_jobs': -1, 'num_leaves': 31, 'objective': 'binary', 'random_state': 7, 'reg_alpha': 7.1000000000000005, 'reg_lambda': 8.1, 'silent': True, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'seed': 7}
# 现阶段模型表现与参数


chosen_final_feature=feat_imp.index[:int(best['feature_num'])]
X_use = X.loc[:, chosen_final_feature]

lgbm_model.fit(X_use, Y, eval_metric=['auc','binary_logloss'])

# train
X,Y = x_train,y_train
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('train，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.6101737528350761, 0.8649091690440008]
# [0.6525369650376067, 0.8862807934521714]
# train，KS值和AUC值分别为[0.31283406586210283, 0.7126621830832252]

# test
X,Y = x_test,y_test
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('test，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.5977482436176529, 0.8566478417014936]
# [0.6898439660019113, 0.9150092752650243]
# [0.31988796801011776, 0.7128375467400838]

# oot1
X,Y = oot1_x,oot1_y
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot1，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.7463033314023115, 0.935975470791987]
# [0.6715034073245776, 0.8911824402804956]
# [0.33712799308429264, 0.7211240343142982]

# oot2
X,Y = oot2_x,oot2_y
preds = lgbm_model.predict_proba(X.loc[:, chosen_final_feature])[:,1]
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('oot2，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.6034645021088599, 0.8592923612387786]
#  [0.4371277291242013, 0.7895184718592859]

# 保存 model 和选取的feature names

path ='/home/bb5/xw/mxg_scores'
lgbm_model.booster_.save_model(path+'/lgbm_v1.model')
chosen_feature_pd=pd.DataFrame(chosen_final_feature)
chosen_feature_pd.to_csv(path+'/lgbm_v1_features.csv',encoding="utf_8_sig",index=False)



# 保存重要性

feat_imp = pd.Series(lgbm_model.feature_importances_,index=X_use.columns)
feat_imp=feat_imp.sort_values(ascending=False)
feat_imp_list = pd.DataFrame(feat_imp)
feat_imp_list2 = feat_imp_list.iloc[0:int(best['feature_num']),:]
feat_imp_list2.to_csv(path+'/chosen_feature_importance_list.csv',index=1)


# 读取保存好的 model 和feature names

input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))



# 出现问题 保存会重新预测会出现问题，score偏离很严重
# lgbm  sumarry
import toad

X,Y = x_train,y_train
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
ks_bucket= toad.metrics.KS_bucket(preds,Y,bucket=10,method='quantile')
ks_bucket.to_csv('train_ks_bucket.csv')
# [0.3163373595235019, 0.7120566742882023]

X,Y = x_test,y_test
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
ks_bucket= toad.metrics.KS_bucket(preds,Y,bucket=10,method='quantile')
ks_bucket.to_csv('test_ks_bucket.csv')
# [0.3187076582949877, 0.7124343237890117]

# oot
X,Y = oot1_x,oot1_y
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
ks_bucket= toad.metrics.KS_bucket(preds,Y,bucket=10,method='quantile')
ks_bucket.to_csv('oot1_ks_bucket.csv')
#[0.3338275233604178, 0.7196792516966626]

X,Y = oot2_x,oot2_y
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
ks_bucket= toad.metrics.KS_bucket(preds,Y,bucket=10,method='quantile')
ks_bucket.to_csv('oot2_ks_bucket.csv')
# [0.4342786366363477, 0.7882866555759933]




"""
********************************************************************
重新提取样本 跑score 为了需求和后面的变量验证
********************************************************************
"""


# 读取数据
sample= pd.read_csv('/home/bb5/xw/mxg_scores/20201123_pg_part.csv',encoding='gbk')
sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)

# 导入模型变量
# 读取保存好的 model 和feature names

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
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
#[0.3301062511292491, 0.7198242221251013]

X,Y = oot2_x,oot2_y
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.4384826660424955, 0.7895857852596792]

X,Y = oot3_x,oot3_y
preds = lgbm_model.predict(X.loc[:, chosen_final_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# [0.33491709884462445, 0.7203960446872948]







