
import pandas as pd

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line,Page
import logging

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
logging.info('hh')

path = '/home/bb5/xw/mxg_scores/monitor'
df = pd.read_csv(path + '/反欺诈变量分布.csv',encoding='gbk')



col = list(df['expression'].unique())

page = Page()
for columns in col:
    tmp = df[df['expression'] == columns ]
    x1 = tmp['value']
    y1 = tmp['t1']
    y2 = tmp['cnt']

    bar = (
        Bar()
            .add_xaxis([ str(i) for i in x1])
            .add_yaxis(
            "cnt",
            y2.to_list(), gap="0%", category_gap="40%",yaxis_index=0,)
            .set_global_opts(
            title_opts=opts.TitleOpts(title=columns),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} %"), position="right", ),)
            .extend_axis(yaxis=opts.AxisOpts(axislabel_opts=opts.LabelOpts(), interval=5
        ))

    )
    line = (
        Line()
            .add_xaxis([ str(i) for i in x1])
            .add_yaxis(
            "due",
            y1.values.round(2),
            yaxis_index=1,)
    )

    overlap_1 = bar.overlap(line)
    page.add(overlap_1)
    logger.info('完成 {}'.format(columns))
page.render("page_anti.html")


