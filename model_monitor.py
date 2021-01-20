



import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
logging.info('hh')



from pyecharts.charts import Bar, Grid, Line,Page
from pyecharts import options as opts
from pyecharts.charts import Bar



# 读取数据
sample = pd.read_csv('/home/bb5/xw/mxg_scores/20210120_pg_v3_old.csv',encoding='gbk')

sample['y']=sample['age'].map(lambda x: 1 if x>=7 else 0)

train = sample.fillna(-999)  # None填充


# 读取保存好的 model 和feature names
path ='/home/bb5/xw/mxg_scores'
input_xgb_features_path=path+'/lgbm_v1_features.csv'
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))
print('chosen_feature: ' ,chosen_feature)


page = Page()
print('模型bin分布 *******************')

# bin 分布
#
# df_bin = pd.read_csv('/home/bb5/xw/mxg_scores/monitor/v3_bin.csv', encoding='gbk')
# bins = ['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7','bin8', 'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15']
# bar_bin = Bar()
# bar_bin.add_xaxis(list(df_bin['apply_time']))
# for bin in bins:
#     bar_bin.add_yaxis(bin, list(df_bin[bin]), stack="stack1")
# bar_bin.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
# bar_bin.set_global_opts(title_opts=opts.TitleOpts(title='v3_bin'),
#                         datazoom_opts = [opts.DataZoomOpts(range_start=0, range_end=100), ],
#                         legend_opts = opts.LegendOpts(pos_left="right", orient='vertical'), )
# bar_bin.render("v3_bin.html")
#
# page.add(bar_bin)


# 模型特征变量分布
print('模型特征变量分布 *******************')


num=0
for feature in chosen_feature:
    num+=1

    feature_value = sample[feature].unique()
    feature_value.sort()
    if len(feature_value)>=15:
        _, bin = pd.qcut(sample[feature], 10, retbins=True, duplicates='drop')
    else: bin = list(feature_value)

    bin = list(bin)
    bin.insert(0, -999999)
    bin.pop()
    bin.append(float('inf'))

    col_list = [str(i) for i in bin]
    train['bin'] = pd.cut(train[feature], bin, right=False, labels=col_list[0:-1])
    tmp = train.groupby(['apply_time', 'bin'])['y'].count()
    tmp = tmp.reset_index()
    tmp = pd.DataFrame(tmp)
    tmp = pd.pivot(tmp, index="apply_time", columns="bin", values="y")
    tmp.columns = tmp.columns.astype(str)
    tmp['total'] = tmp.apply(lambda x: x.sum(), axis=1)

    col_list = tmp.columns.to_list()
    bar = Bar()
    bar.add_xaxis(list(tmp.index))
    for i in range(len(col_list[0:-1])):
        bar.add_yaxis(col_list[i], list(tmp[col_list[i]] / tmp['total']), stack="stack1")

    bar.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    bar.set_global_opts(title_opts=opts.TitleOpts(title=feature),
                        datazoom_opts=[opts.DataZoomOpts(range_start=0,range_end=100), ],
                        #toolbox_opts=opts.ToolboxOpts(),
                        legend_opts=opts.LegendOpts(pos_left="right",orient='vertical'),)

    page.add(bar)
    logger.info('数量 {}  完成 {}'.format(num,feature))

page.render("v3_old.html")



