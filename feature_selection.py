



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
from lgbm_tuner import py_overlap

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line,Page



# 读取数据
sample01 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part1.csv',encoding='gbk')
sample02 = pd.read_csv('/home/bb5/xw/mxg_scores/20201117_pg_part2.csv',encoding='gbk')
sample =pd.merge(sample01,sample02,how='inner',left_on='biz_id',right_on='biz_ids')

del sample01,sample02

sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)



# train+ oot
train=sample[sample['apply_time']<='2020-10-09']
oot=sample[(sample['apply_time']>='2020-10-10')]



# 读取保存好的 model 和feature names
path ='/home/bb5/xw/mxg_scores'
input_xgb_features_path=path+'/lgbm_v1_features.csv'
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))


# def py_overlap(feature,cut_list,train_cnt_rate,oot_cnt_rate,train_due_rate,oot_due_rate):
#     bar = (
#         Bar()
#             .add_xaxis([ str(i) for i in cut_list])
#             .add_yaxis(
#             "train_rate",
#             list(train_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
#             yaxis_index=0,)
#             .add_yaxis(
#             "oot_rate",
#             list(oot_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
#             yaxis_index=0,)
#             .set_global_opts(
#             title_opts=opts.TitleOpts(title=feature),
#             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %"), position="right", ),
#         )
#
#     )
#
#     line = (
#         Line()
#             .add_xaxis([ str(i) for i in cut_list])
#             .add_yaxis(
#             "train_due",
#             train_due_rate.values.round(2),
#             yaxis_index=0,)
#             .add_yaxis(
#             "oot_due",
#             oot_due_rate.values.round(2),
#             yaxis_index=0,)
#     )
#
#     overlap_1 = bar.overlap(line)
#     overlap_1.render_notebook()
#     return  overlap_1

page = Page()
for feature in chosen_feature:
    logger.info(feature)
    uniq = train[feature].unique()
    if np.isnan(np.min(uniq)):
        uniq = uniq[~np.isnan(uniq)]

    uniq = sorted(uniq)
    cut_list = [ uniq[int(len(uniq)*i/8)]  for i in range(8)] + [uniq[-1]+1]  # 中位数，前后两端都要包含
    logger.info(cut_list)
    train['cut_bins'] = pd.cut(train[feature], cut_list,right=False,duplicates='drop')
    train_cnt_rate = train.groupby('cut_bins')['y'].count() / len(train)+0.001
    train_due_rate = train.groupby('cut_bins')['y'].mean()+0.001

    oot['cut_bins'] = pd.cut(oot[feature], cut_list,right=False,duplicates='drop')
    oot_cnt_rate = oot.groupby('cut_bins')['y'].count() / len(oot)+0.001
    oot_due_rate = oot.groupby('cut_bins')['y'].mean()+0.001

    overlap = py_overlap(feature, cut_list, train_cnt_rate, oot_cnt_rate, train_due_rate, oot_due_rate)
    page.add(overlap)
    logger.info('完成 {}'.format(feature))

page.render('page.html')
logger.info('完成 {}'.format('page.html'))



# 缺失值统计




logger.info(train.loc[:,chosen_feature].isna().sum()/len(train))
logger.info(oot.loc[:,chosen_feature].isna().sum()/len(train))

