








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
sample = pd.read_csv('/home/bb5/xw/mxg_scores/feature_check/20201218_pg_online_v1.csv',encoding='gbk')

sample['y']=sample['age_value']
train = sample.fillna(-999)  # None填充
train['apply_time']=pd.to_datetime(train['apply_time'],format="%Y-%m-%d")



# # 读取保存好的 model 和feature names
# path ='/home/bb5/xw/mxg_scores'
# input_xgb_features_path=path+'/lgbm_v1_features.csv'
# chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
# chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))
# print('chosen_feature: ' ,chosen_feature)



a = train.columns.to_list()





page = Page()
for feature in a[7:]:
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
                        datazoom_opts=[opts.DataZoomOpts(), ],
                        #toolbox_opts=opts.ToolboxOpts(),
                        legend_opts=opts.LegendOpts(pos_left="right",orient='vertical'),)

    page.add(bar)
    logger.info('完成 {}'.format(feature))

page.render("page_simple_layout.html")






