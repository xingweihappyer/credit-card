



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


# train+oot+进件样本

train=sample[ (sample['apply_time']<='2020-10-09')]
train_y = train['y']
train_x = train
print('train 样本数',train_y.count())
print('train 样本逾期率',train_y.sum()/train_y.count())


oot=sample[ (sample['apply_time']>='2020-10-10')]
oot_y = oot['y']
oot_x = oot
print('oot 样本数',oot_y.count())
print('oot 样本逾期率',oot_y.sum()/oot_y.count())


X,Y = train_x,train_y
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))

X,Y = oot_x,oot_y
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))


# 设置  factor  offset  score=int(log((1-pvalue_off)/pvalue_off)*factor+offset);
factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
logger.info('factor:{} offset:{}'.format(factor,offset))

# train 打分

preds = lgbm_model.predict(train.loc[:, chosen_feature])
preds = pd.DataFrame({'preds':preds})
train = train.reset_index()
train_df = pd.merge(train,preds,left_index=True, right_index=True)
train_df['score'] =  (np.log((1 - train_df['preds'])/train_df['preds']) * factor + offset).round(0)
train_ks_bucket= KS_bucket(train_df['score'],train_df['y'],bucket=15,method='quantile')
train_ks_bucket[['min', 'max', 'bads', 'goods', 'total', 'bad_rate']].to_csv('/home/bb5/xw/mxg_scores/strategy/train_ks_bucket.csv')
# 获取 score 分割点 ，选择最大的，就要填充最小的和替换最大的

cut = train_ks_bucket['max'].to_list()
cut.insert(0,float("-inf"))
cut.pop()
cut.append(float("inf"))
logger.info('cut 分割点 {}'.format(cut))



def  model_score(oot):
    preds = lgbm_model.predict(oot.loc[:, chosen_feature])
    preds = pd.DataFrame({'preds': preds})
    oot = oot.reset_index()
    oot_df = pd.merge(oot, preds, left_index=True, right_index=True)
    oot_df['score'] = (np.log((1 - oot_df['preds']) / oot_df['preds']) * factor + offset).round(0)
    oot_df['bin'] = pd.cut(oot_df['score'], cut, right=True)

    oot_cnt = oot_df.groupby(['bin'])['y'].count()
    oot_cntrate = oot_df.groupby(['bin'])['y'].count() / len(oot_df)
    oot_badrate = oot_df.groupby(['bin'])['y'].sum() / oot_df.groupby(['bin'])['y'].count()

    oot_ks_bucket = pd.DataFrame({'oot_cnt': oot_cnt, 'oot_cntrate': oot_cntrate, 'oot_badrate': oot_badrate})
    logger.info(oot_ks_bucket)
    return oot_ks_bucket


"""
oot 打分
"""
oot_ks_bucket = model_score(oot)
oot_ks_bucket.to_csv('/home/bb5/xw/mxg_scores/strategy/oot_ks_bucket.csv')
train_ks_bucket = model_score(train)
train_ks_bucket.to_csv('/home/bb5/xw/mxg_scores/strategy/train_ks_bucket.csv')
"""
进件样本打分
"""

# 读取数据
apply= pd.read_csv('/home/bb5/xw/mxg_scores/20201125_pg_apply.csv',encoding='gbk')
apply['y']=apply['overdue_days'].map(lambda x: 1 if x>=7 else 0)
apply_ks_bucket = model_score(apply)
apply_ks_bucket.to_csv('/home/bb5/xw/mxg_scores/strategy/apply_ks_bucket.csv')




#
#
# """
# 对比之前的模型效果  auc=0.69
# """
#
# # 读取数据
# wml= pd.read_csv('/home/bb5/xw/mxg_scores/20201125_pg_wml.csv',encoding='gbk')
# wml['y']=wml['overdue_days'].map(lambda x: 1 if x>=7 else 0)
#
# wml_y = wml['y']
# wml_x = wml
#
# X,Y = wml_x,wml_y
# preds = lgbm_model.predict(X.loc[:, chosen_feature])
# ks = KS(preds,Y)
# auc = AUC(preds,Y)
# print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# # [0.33491709884462445, 0.7203960446872948]
#
#
#
# """
# 对比之前的模型效果  0413  3411 效果计算  [0.3007182133666338, 0.7025405389244967]
# """
#
# # 读取数据
# wml= pd.read_csv('/home/bb5/xw/mxg_scores/20201125_pg_wml3411.csv',encoding='gbk')
# wml['y']=wml['overdue_days'].map(lambda x: 1 if x>=7 else 0)
#
# wml_y = wml['y']
# wml_x = wml
#
# X,Y = wml_x,wml_y
# preds = lgbm_model.predict(X.loc[:, chosen_feature])
# ks = KS(preds,Y)
# auc = AUC(preds,Y)
# print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# # [0.33491709884462445, 0.7203960446872948]
#
#
# """
# 0413  3411 效果计算 [0.34751247041940625, 0.7180073469494138]
# """
#
# # 读取数据
# wml0413= pd.read_csv('/home/bb5/xw/mxg_scores/wml0413.csv')
# wml_test = pd.merge(wml,wml0413,left_on='biz_id',right_on='biz_ids')
#
#
# preds = wml_test['score']
# Y = wml_test['y']
#
# ks = KS(preds,Y)
# auc = AUC(preds,Y)
# print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# # [0.33491709884462445, 0.7203960446872948]




# 策略方法2 ，重通过率，先进件封箱，在映射放款


# 导入模型变量
# 读取保存好的 model 和feature names
path ='/home/bb5/xw/mxg_scores'
input_model_path=path+'/lgbm_v1.model'
input_xgb_features_path=path+'/lgbm_v1_features.csv'


lgbm_model = lgb.Booster(model_file=input_model_path)
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))

# 读取数据
apply= pd.read_csv('/home/bb5/xw/mxg_scores/20201125_pg_apply.csv',encoding='gbk')
apply['y']=apply['overdue_days'].map(lambda x: 1 if x>=7 else 0)

# 设置  factor  offset  score=int(log((1-pvalue_off)/pvalue_off)*factor+offset);
factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
logger.info('factor:{} offset:{}'.format(factor,offset))


apply['preds'] = lgbm_model.predict(apply.loc[:, chosen_feature])
apply['score'] =  (np.log((1 - apply['preds'])/apply['preds']) * factor + offset).round(0)
apply_ks_bucket= KS_bucket(apply['score'],apply['y'],bucket=20,method='quantile')
apply_ks_bucket[['min', 'max', 'bads', 'goods', 'total', 'bad_rate']].to_csv('/home/bb5/xw/mxg_scores/strategy/apply_ks_bucket.csv')
# 获取 score 分割点 ，选择最大的，就要填充最小的和替换最大的

cut = apply_ks_bucket['max'].to_list()
cut.insert(0,float("-inf"))
cut.pop()
cut.append(float("inf"))
logger.info('cut 分割点 {}'.format(cut))

def  model_score(oot):
    preds = lgbm_model.predict(oot.loc[:, chosen_feature])
    preds = pd.DataFrame({'preds': preds})
    oot = oot.reset_index()
    oot_df = pd.merge(oot, preds, left_index=True, right_index=True)
    oot_df['score'] = (np.log((1 - oot_df['preds']) / oot_df['preds']) * factor + offset).round(0)
    oot_df['bin'] = pd.cut(oot_df['score'], cut, right=True)

    oot_cnt = oot_df.groupby(['bin'])['y'].count()
    oot_cntrate = oot_df.groupby(['bin'])['y'].count() / len(oot_df)
    oot_badrate = oot_df.groupby(['bin'])['y'].sum() / oot_df.groupby(['bin'])['y'].count()

    oot_ks_bucket = pd.DataFrame({'oot_cnt': oot_cnt, 'oot_cntrate': oot_cntrate, 'oot_badrate': oot_badrate})
    logger.info(oot_ks_bucket)
    return oot_ks_bucket



"""
oot 打分
"""
train_ks_bucket = model_score(train)
train_ks_bucket.to_csv('/home/bb5/xw/mxg_scores/strategy/train_ks_bucket.csv')
oot_ks_bucket = model_score(oot)
oot_ks_bucket.to_csv('/home/bb5/xw/mxg_scores/strategy/oot_ks_bucket.csv')

"""
进件样本打分
"""