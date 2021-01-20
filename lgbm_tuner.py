

# 使用网格搜索方法 确定lgbm的参数值
from lightgbm import LGBMClassifier
import lightgbm as lgb
from toad.metrics import KS, F1, AUC

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import logging
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line
# 日志管理
logger_name = "lgbm"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)
logger.info('test')

# 使用网格搜索方法 确定lgbm的参数值

def init_feature(x_train, y_train):
    X, Y = x_train, y_train
    lgbm_model = LGBMClassifier(
    learning_rate=0.05,
    n_estimators=500,
    max_depth=4,
    min_split_gain=0.01,
    min_child_samples=20,
    subsample=1,
    colsample_bytree=1,
    importance_type='split',
    objective='binary',
    random_state=7)

    lgbm_param = lgbm_model.get_params()
    lgbm_train = lgb.Dataset(X, Y)
    lgbm_param.pop('silent')
    lgbm_param.pop('n_estimators')

    '''使用交叉验证的方式确定最优的树数量'''
    cvresult = lgb.cv(lgbm_param, lgbm_train, num_boost_round=100, nfold=4, metrics=['auc','binary_logloss'], early_stopping_rounds=100)
    best_n_estimators = len(cvresult['auc-mean'])
    print('确定最优的树数量', best_n_estimators)

    lgbm_model.set_params(n_estimators=best_n_estimators)
    # lgbm_model.fit(X,Y,eval_metric='auc')
    lgbm_model.fit(X, Y, eval_metric=['auc', 'binary_logloss'])

    feat_imp = pd.Series(lgbm_model.feature_importances_, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False)

    valid_feature_num = len(np.where(feat_imp > 0)[0])  # 有效变量是有feature_importance的变量（在lgbm树模型中有贡献的变量，其他的变量没有用到）
    print('有效变量数为{0}个'.format(valid_feature_num))

    return feat_imp



# from sklearn import svm, datasets
# from sklearn.model_selection import GridSearchCV
# iris = datasets.load_iris()
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC()
# clf = GridSearchCV(svc, parameters,scoring='roc_auc',cv=4)
# clf.fit(iris.data[0:100], iris.target[0:100])
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

def grid_search(parameters,x_train, x_test, y_train, y_test):
    best_auc = 0
    for num_leaves in parameters['num_leaves']:
        for max_depth in  parameters['max_depth']:
            for learning_rate in parameters['learning_rate']:
                for n_estimators in parameters['n_estimators']:
                        # for min_split_gain in parameters['min_split_gain']:
                        #     for min_child_weight in parameters['min_child_weight']:
                        #         for min_child_samples in parameters['min_child_samples']:
                        #             for subsample in parameters['subsample']:
                        #                 for colsample_bytree in parameters['colsample_bytree']:
                        #                     for reg_alpha in parameters['reg_alpha']:
                        #                         for reg_lambda in parameters['reg_lambda']:
                                                    # logger.info('{},{},{},{},{},{},{},{},{},{},{}'.format(num_leaves,
                                                    #                                             max_depth,
                                                    #                                             learning_rate,
                                                    #                                             n_estimators,
                                                    #                                             min_split_gain,
                                                    #                                             min_child_weight,
                                                    #                                             min_child_samples,
                                                    #                                             subsample,
                                                    #                                             colsample_bytree,
                                                    #                                             reg_alpha,
                                                    #                                             reg_lambda
                                                    #                                             ))
                                                    lgbm_model = LGBMClassifier(
                                                        num_leaves=num_leaves,
                                                        max_depth=max_depth,
                                                        learning_rate=learning_rate,
                                                        n_estimators=n_estimators,
                                                        # min_split_gain=min_split_gain,
                                                        # min_child_weight=min_child_weight,
                                                        # min_child_samples=min_child_samples,
                                                        # subsample=subsample,
                                                        # colsample_bytree=colsample_bytree,
                                                        # reg_alpha=reg_alpha,
                                                        # reg_lambda=reg_lambda,
                                                        importance_type='split',
                                                        objective='binary',
                                                        random_state=7)

                                                    lgbm_model.fit(x_train, y_train, eval_metric='auc')
                                                    preds = lgbm_model.predict(x_test)
                                                    auc = AUC(preds,y_test)
                                                    if auc>best_auc:
                                                        best_auc = auc
                                                        logger.info('test auc:{}'.format(best_auc))
                                                        best_para = lgbm_model.get_params()
    return best_para



def py_overlap(feature,cut_list,train_cnt_rate,oot_cnt_rate,train_due_rate,oot_due_rate):
    bar = (
        Bar()
            .add_xaxis([ str(i) for i in cut_list])
            .add_yaxis(
            "train_rate",
            list(train_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
            yaxis_index=0,)
            .add_yaxis(
            "oot_rate",
            list(oot_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
            yaxis_index=0,)
            .set_global_opts(
            title_opts=opts.TitleOpts(title=feature),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %"), position="right", ),
        )

    )

    line = (
        Line()
            .add_xaxis([ str(i) for i in cut_list])
            .add_yaxis(
            "train_due",
            train_due_rate.values.round(2),
            yaxis_index=0,)
            .add_yaxis(
            "oot_due",
            oot_due_rate.values.round(2),
            yaxis_index=0,)
    )

    overlap_1 = bar.overlap(line)
    overlap_1.render_notebook()
    return  overlap_1





def xw_bar(feature,cut_list,train_cnt_rate,oot_cnt_rate):
    bar = (
        Bar()
            .add_xaxis(cut_list)
            .add_yaxis(
            "train",
            list(train_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
            yaxis_index=0,)
            .add_yaxis(
            "oot",
            list(oot_cnt_rate.values.round(2)), gap="0%", category_gap="40%",
            yaxis_index=0,)
            .set_global_opts(
            title_opts=opts.TitleOpts(title=feature),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %"), position="right", ),
        )

    )
    return bar



def xw_line(feature,cut_list,train_due_rate,oot_due_rate):
    line = (
        Line()
            .add_xaxis(cut_list)
            .add_yaxis(
            "train",
            list(train_due_rate.values.round(2)),
            yaxis_index=0,)
            .add_yaxis(
            "oot",
            list(oot_due_rate.values.round(2)),
            yaxis_index=0,)
            .set_global_opts(
            title_opts=opts.TitleOpts(title=feature),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value}"), position="right", ),
        )

    )
    return line

