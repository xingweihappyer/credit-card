

# 以下图像均需在jupyter运行




import pyecharts.options as opts
from pyecharts.charts import Funnel
from pyecharts.charts import Bar, Grid, Line,Page

# 漏斗图
x_data = ["下载", "注册", "认证", "进件", "反欺诈","模型"]
y_data = [100, 46, 36, 26, 21,12]


data = [[x_data[i], y_data[i]] for i in range(len(x_data))]
c = (
    Funnel()
    .add(
        series_name="",
        data_pair=data,
        gap=2,
        tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{a} <br/>{b} : {c}%"),
        label_opts=opts.LabelOpts(is_show=True, position="inside",font_size=14,formatter='{b} : {c}%'),
        itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1),)
    .set_global_opts(title_opts=opts.TitleOpts(title="7天 首贷"),
)
)

c.render_notebook()



# lift 
def lift(df,title):
    line = (
        Line()
            .add_xaxis(df['bin'].to_list())
            .add_yaxis("T1", (df['t1'] * 100).round(0), )
            .add_yaxis("T3", (df['t3'] * 100).round(0), )
            .add_yaxis("T5", (df['t5'] * 100).round(0), )
            .add_yaxis("T7", (df['t7'] * 100).round(0), )

            .set_series_opts(label_opts=opts.LabelOpts())
            .set_global_opts(title_opts=opts.TitleOpts(title=title),
                             yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(formatter="{value} "),
                                                      position="right"))
    )
    return line





import pandas as pd
# path ='/home/bb5/xw/mxg_scores/report'
path =r'E:\funbox_reco\funbox_git\mxg_scores\report'
df = pd.read_csv(path + '/复贷lift.csv',encoding='gbk')
lift = lift(df,'7天 复贷模型 lift')
lift.render_notebook()



# 进件
import pandas as pd
path = r'E:\Mexico墨西哥\墨西哥风控\report'
df = pd.read_csv(path + '\首贷进件7天.csv',encoding='gbk')


bar = (
    Bar()
        .add_xaxis(df['申请日'].to_list())
        .add_yaxis("申请数",
        df['申请数'].to_list(), gap="0%", category_gap="40%",yaxis_index=0)
        .set_global_opts(title_opts=opts.TitleOpts(title='7天 首贷进件'),)
        .extend_axis(
        yaxis=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(), interval=5
        )
    )

)
bar.render_notebook()


line = (
    Line()
        .add_xaxis(df['申请日'].to_list())
        .add_yaxis("通过率",df['通过率'],yaxis_index=1, )
        .add_yaxis("拒绝率",df['拒绝率'],yaxis_index=1, )
        .add_yaxis("反欺诈拒绝率",df['反欺诈拒绝率'],yaxis_index=1, )
        .add_yaxis("风控拒绝率",df['风控拒绝率'],yaxis_index=1, )
        .set_series_opts(label_opts=opts.LabelOpts())
)
line.render_notebook()

overlap_1 = bar.overlap(line)
overlap_1.render_notebook()











