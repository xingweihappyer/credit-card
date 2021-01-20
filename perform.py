# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:52:35 2018

@author: zhaifeifei
"""
import pandas as pd
import numpy as np
import matplotlib.dates as mdate
import datetime
from sklearn.metrics import roc_curve,auc
import xlwt
from xlwt import *
import matplotlib.pyplot as plt
# from pyecharts import Timeline, Line, Page, Bar, Overlap
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei'] #设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  #设置正常显示字符
warnings.filterwarnings('ignore')
import os

## 计算K-S
def cal_ks(point,Y,section_num=10):
    
    Y=pd.Series(Y)
    sample_num=len(Y)
    
    bad_percent=np.zeros([section_num,1])
    good_percent=np.zeros([section_num,1])
    
    point=pd.DataFrame(point)
    sorted_point=point.sort_values(by=0,ascending=False)   #负值从大到小，正常值从小到大
    total_bad_num=len(np.where(Y==1)[0])
    total_good_num=len(np.where(Y==0)[0])
    
    for i in range(0,section_num):
        split_point=sorted_point.iloc[int(round(sample_num*(i+1)/section_num))-1]
        position_in_this_section=np.where(point>=split_point)[0]
        bad_percent[i]=len(np.where(Y.iloc[position_in_this_section]==1)[0])/total_bad_num   #通过分割点，不分箱，同样的累计值效果
        good_percent[i]=len(np.where(Y.iloc[position_in_this_section]==0)[0])/total_good_num
        
    ks_value=np.abs(bad_percent-good_percent)

    return ks_value,bad_percent,good_percent

## 计算PSI
def PSI(score_train,score_test,section_num=10):
    score_train=pd.DataFrame(score_train)
    score_test=pd.DataFrame(score_test)
    
    total_train_num=len(score_train)
    total_test_num=len(score_test)
    
    sorted_score_train=score_train.sort_values(by=0)    
    
    PSI_value=0
    
    for i in range(0,section_num):
        lower_bound=sorted_score_train.iloc[int(round(total_train_num*(i)/section_num))]
        higher_bound=sorted_score_train.iloc[int(round(total_train_num*(i+1)/section_num))-1]
        score_train_percent=len(np.where((score_train>=lower_bound)&(score_train<=higher_bound))[0])/total_train_num
        score_test_percent=len(np.where((score_test>=lower_bound)&(score_test<=higher_bound))[0])/total_test_num
        
        PSI_value +=(score_test_percent-score_train_percent)*np.log(score_test_percent/score_train_percent)
        
    return PSI_value

## 空缺值不做处理的woe计算方法
def WOE(data, dim, bond_num=[],left_close=1):

    m = data.shape[0]
    X = data[:, dim]
    y = data[:, -1]
    tot_bad = np.sum(y == 1)
    tot_good = np.sum(y == 0)
    data = np.column_stack((X.reshape(m, 1), y.reshape(m, 1)))
    cnt_bad = []
    cnt_good = []

    index = bond_num
    bucket_num = bond_num.shape[0] - 1
    data_bad = data[data[:, 1] == 1, 0]
    data_good = data[data[:, 1] == 0, 0]
    cnt_bad.append(sum(np.isnan(data_bad)))
    cnt_good.append(sum(np.isnan(data_good)))
    data_bad_nonan =data_bad[~np.isnan(data_bad)]
    data_good_nonan =data_good[~np.isnan(data_good)]
    
    ## 评分卡每箱的格式是左闭右开还是左开右闭
    left_close =1 #左边闭合
    if left_close ==1:
        eps_left =1e-8
        eps_right =0
    else:
        eps_left =0
        eps_right =1e-8   
        
    eps = 1e-8
    
    for i in range(bucket_num):
        if i < bucket_num - 1:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad_nonan > index[i] - eps_left, data_bad_nonan < index[i + 1]+eps_right)))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good_nonan > index[i] - eps_left, data_good_nonan < index[i + 1]+eps_right)))
        else:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad_nonan > index[i] - eps, data_bad_nonan < index[i + 1] + eps)))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good_nonan > index[i] - eps, data_good_nonan < index[i + 1] + eps)))
    bond = np.array(index)
    cnt_bad = np.array(cnt_bad)
    cnt_good = np.array(cnt_good)
    
    #对完美分箱增加一个虚拟样本，保证有woe值
    cnt_bad[cnt_bad==0]+=1
    cnt_good[cnt_good==0]+=1
        
    length = cnt_bad.shape[0]
    for i in range(length):
        j = length - i - 1
        if j != 0:
            if cnt_bad[j] == 0 or cnt_good[j] == 0:
                cnt_bad[j - 1] += cnt_bad[j]
                cnt_good[j - 1] += cnt_good[j]
                cnt_bad = np.append(cnt_bad[:j], cnt_bad[j + 1:])
                cnt_good = np.append(cnt_good[:j], cnt_good[j + 1:])
                bond = np.append(bond[:j], bond[j + 1:])
    if cnt_bad[0] == 0 or cnt_good[0] == 0:
        cnt_bad[1] += cnt_bad[0]
        cnt_good[1] += cnt_good[0]
        cnt_bad = cnt_bad[1:]
        cnt_good = cnt_good[1:]
        bond = np.append(bond[0], bond[2:])
    woe = np.log((cnt_bad / tot_bad) / (cnt_good / tot_good))
    IV = ((cnt_bad / tot_bad) - (cnt_good / tot_good)) * woe
    IV_tot = np.sum(IV)
    bond_str = []
    for b in bond:
        bond_str.append(str(b))
    box_num=  cnt_bad+ cnt_good
    bad_rate=cnt_bad/box_num
    
    return IV_tot, IV, woe, bond, box_num, bad_rate

## 空缺值填充为-99的woe计算方法    
def WOE2(data, dim, bond_num=[]):

    m = data.shape[0]
    X = data[:, dim]
    y = data[:, -1]
    tot_bad = np.sum(y == 1)
    tot_good = np.sum(y == 0)
    data = np.column_stack((X.reshape(m, 1), y.reshape(m, 1)))
    cnt_bad = []
    cnt_good = []
    index = bond_num
    bucket_num = bond_num.shape[0] - 1
    data_bad = data[data[:, 1] == 1, 0]
    data_good = data[data[:, 1] == 0, 0]
    eps = 1e-8
    
    for i in range(bucket_num):
        if i < bucket_num - 1:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1])))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1])))
        else:
            cnt_bad.append(1.0 * np.sum(np.bitwise_and(data_bad > index[i] - eps, data_bad < index[i + 1] + eps)))
            cnt_good.append(1.0 * np.sum(np.bitwise_and(data_good > index[i] - eps, data_good < index[i + 1] + eps)))
    bond = np.array(index)
    cnt_bad = np.array(cnt_bad)
    cnt_good = np.array(cnt_good)
    
    #对完美分箱增加一个虚拟样本，保证有woe值
    cnt_bad[cnt_bad==0]+=1
    cnt_good[cnt_good==0]+=1
        
    length = cnt_bad.shape[0]
    for i in range(length):
        j = length - i - 1
        if j != 0:
            if cnt_bad[j] == 0 or cnt_good[j] == 0:
                cnt_bad[j - 1] += cnt_bad[j]
                cnt_good[j - 1] += cnt_good[j]
                cnt_bad = np.append(cnt_bad[:j], cnt_bad[j + 1:])
                cnt_good = np.append(cnt_good[:j], cnt_good[j + 1:])
                bond = np.append(bond[:j], bond[j + 1:])
    if cnt_bad[0] == 0 or cnt_good[0] == 0:
        cnt_bad[1] += cnt_bad[0]
        cnt_good[1] += cnt_good[0]
        cnt_bad = cnt_bad[1:]
        cnt_good = cnt_good[1:]
        bond = np.append(bond[0], bond[2:])
    woe = np.log((cnt_bad / tot_bad) / (cnt_good / tot_good))
    IV = ((cnt_bad / tot_bad) - (cnt_good / tot_good)) * woe
    IV_tot = np.sum(IV)
    box_num=  cnt_bad+ cnt_good
    bad_rate=cnt_bad/box_num
    
    return IV_tot, IV, woe, bond, box_num, bad_rate


## 等频分箱,输出分割点
def SplitData(df, col, numOfSplit, special_attribute=[]):
    '''
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    '''
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = N//numOfSplit
    splitPointIndex = [i*n for i in range(1,numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint

## 单调性检验'''
def BadRateMonotone(this_woe,if_tolerate_inflextion):

    woe =this_woe
    woe_direction=np.sign(woe[1:]-woe[:-1])
    direction_change_position=woe_direction[1:]-woe_direction[:-1]
    direction_change_times=len(np.where(np.abs(direction_change_position)==2)[0]) #direction_change_times为转向次数

   ## 若第一个转折点发生在缺失值分箱和第一个正常值分箱处，则该转折点不计数'''
    if direction_change_times>=1:
        
        if  np.where(np.abs(direction_change_position)==2)[0][0]==0: ## 转折点发生在缺失值和第一个正常值之间不计数
            direction_change_times-=1
           
    ################        
    if direction_change_times<1+if_tolerate_inflextion:   #若转折点少于两个，则满足WOE分箱条件
        return True
    else:
        return False


## 统计十等分每段的好坏人数
def cal_good_bad(point,Y,section_num=10):
    Y=pd.Series(Y)
    sample_num=len(Y)
    
    bad_num=np.zeros([section_num,1])
    good_num=np.zeros([section_num,1])
    
    point=pd.DataFrame(point)
    sorted_point=point.sort_values(by=0)
    split_points =[]
    for i in range(0,section_num):
        split_point=sorted_point.iloc[int(round(sample_num*(i+1)/section_num))-1]
        split_points.append(split_point)
        if i==0:
            position_in_this_section=np.where(point<=split_point)[0]
        else:
            split_point_before=sorted_point.iloc[int(round(sample_num*(i)/section_num))-1]
            position_in_this_section=np.where((point<=split_point)&(point>split_point_before))[0]
            
        bad_num[i]=len(np.where(Y.iloc[position_in_this_section]==1)[0])
        good_num[i]=len(np.where(Y.iloc[position_in_this_section]==0)[0])
        
    return bad_num,good_num,split_points

####################
''' 2.作变量长期趋势'''
def var_trend(data, true_y, var_label, titel, outputh,card_number=0):

    total_columns = data.columns
    woe_columns = [i for i in total_columns if i.endswith(var_label)]
    page  = Page()
    page2 = Page()
    for item in woe_columns:
        df_new = data[['cid', item, true_y, 'add_day']]
        df_stand = df_new.groupby(['add_day', item])[true_y].mean()
        df_stand_1 = df_stand.reset_index()
        df_stand_1 = df_stand_1.set_index('add_day')
        df_table = pd.pivot_table(data=df_stand_1, index='add_day', columns=item, values=true_y)

        df_date_stand = data[['add_day', 'cid', item]].groupby(['add_day', item]).count()
        df_date_stand = df_date_stand.reset_index()
        df_date_stand_new = pd.pivot_table(data=df_date_stand, index='add_day', columns=item, values='cid')
        df_date_stand_new = df_date_stand_new.dropna(thresh=3, axis=1)
        df_date_stand_new = df_date_stand_new.fillna(0)
        df_date_stand_new_2 = df_date_stand_new.apply(lambda x: x / sum(x), axis=1)
#
        Line_1 = Line(item, width=1400, height=500)
        for its in df_table.columns:
            Line_1.add("{}".format(its), df_table.index, df_table[its], is_smooth=True, line_width=2, mark_line=["average"],
                       datazoom_range=[0, 100],tooltip_trigger='axis', legend_top='5%', legend_text_size="13", is_datazoom_show=True)
        page.add(Line_1)
#
        bar = Bar(item, width=1400, height=500)
        for var in df_date_stand_new_2.columns:
            bar.add("{}".format(var), list(df_date_stand_new_2.index), list(df_date_stand_new_2[var]), is_stack=True,
                    yaxis_max=1, is_datazoom_show=True, datazoom_range=[0, 100],
                    is_smooth=True, legend_orient='vertical', legend_pos='90%', legend_top='30%', is_toolbox_show=False,
                    is_more_utils=False, legend_text_size="10")
        page2.add(bar)
    filename1 = '{}_{}评分卡变量趋势图'.format(titel, card_number)
    page.render( outputh + filename1 + '.html')

    filename2 = '{}_{}评分卡变量分Bin图'.format(titel, card_number)
    page2.render(outputh + filename2 + '.html')
    return filename1, filename2


def cal_ratio(data, var, class_label,target):

    if class_label == 1:
        data = data.loc[data['tto_label']==1]
        x1 = 'train_pcnt'
        x2 = 'train_bad_rate'
    elif class_label ==2:
        data = data.loc[data['tto_label']==2]
        x1 = 'test_pcnt'
        x2 = 'test_bad_rate'
    else:
        data =  data.loc[data['tto_label']==3]
        x1 = 'oot_pcnt'
        x2 = 'oot_bad_rate'
    data_new = data[[var,target]].groupby([var]).count().reset_index()
    data_new['total'] = data_new[target].sum()
    data_new[x1] = data_new[target]/data_new['total']
    data_new[x1] = data_new[x1].map(lambda x: round(x * 100, 2))
    data_bad = data[[var,target]].groupby([var])[target].sum().reset_index()
    data_bad.columns = [var, 'bad']
    data_final = pd.merge(data_new, data_bad, on=var)
    data_final[x2] = data_final['bad'] / data_final[target]
    data_final[x2] = data_final[x2].map(lambda x: round(x * 100, 2))
    data_final = data_final[[var, x1, x2]]
    return data_final


def train_test_oot(data, target, tto_label, filename,outputh):
    
    total_columns = [i.lower() for i in data.columns]
    var_columns = [i for i in total_columns if (i.endswith('_label')) &(i!='tto_label')]
    data = data.loc[(data[target]==1)|(data[target]==0)]
    page = Page()
    for var in var_columns:
        data_var = data[['cid',target,var,tto_label]]
        train = cal_ratio(data_var, var, 1, target)
        test = cal_ratio(data_var, var, 2, target)
        oot = cal_ratio(data_var, var, 3, target)
        Line_1 = Line('{} 变量分BIN占比及bad_rate'.format(var), width=1200, height=500)

        Line_1.add("train_{}".format(var), train[var], train.iloc[:, 2], is_stack=False, is_datazoom_show=True,is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(194,53,49)"],
                  datazoom_range=[0, 100],legend_top='7%', is_toolbox_show=False, is_more_utils=False, legend_text_size="12")
        Line_1.add("test_{}".format(var), test[var], test.iloc[:, 2], is_stack=False, is_datazoom_show=True,
                  datazoom_range=[0, 100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(102,139,139)"],
                  legend_top='7%', is_toolbox_show=False, is_more_utils=False, legend_text_size="12")
        Line_1.add("oot_{}".format(var), oot[var], oot.iloc[:, 2], is_stack=False, is_datazoom_show=True,
                  datazoom_range=[0, 100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(47,69,84)"],
                  legend_top='7%', is_toolbox_show=False, is_more_utils=False, legend_text_size="12")


        Bar_1 = Bar('{} 变量分BIN占比及bad_rate'.format(var), width=1200, height=500)
        Bar_1.add("train_{}".format(var), train[var], train.iloc[:,1],is_stack=False, is_datazoom_show=True,datazoom_range=[0,100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(194,53,49)"],
                legend_top='7%',is_toolbox_show=False, is_more_utils=False,legend_text_size="12")
        Bar_1.add("test_{}".format(var), test[var], test.iloc[:,1],is_stack=False, is_datazoom_show=True,datazoom_range=[0,100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(102,139,139)"],
                legend_top='7%',is_toolbox_show=False, is_more_utils=False,legend_text_size="12")
        Bar_1.add("oot_{}".format(var), oot[var], oot.iloc[:,1],is_stack=False, is_datazoom_show=True,datazoom_range=[0,100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(47,69,84)"],
                legend_top='7%',is_toolbox_show=False, is_more_utils=False,legend_text_size="12")

        overlap1 = Overlap(width=1300, height=400)
        overlap1.add(Bar_1)
        overlap1.add(Line_1, yaxis_index=1, is_add_yaxis=True)
        page.add(overlap1)
    page.render(outputh + filename+ '_'+'train_test_oot_bin.html')

def cal_ratio2(data, var, class_label,target):
    # data = data.loc[~data[target].isnull()]
    """
    data =data_var
    class_label =1
    """
    if class_label == 1:
        data = data.loc[data['tto_label']==1]
        x1 = 'train_pcnt'
        x2 = 'train_bad_rate'
    else:
        data =  data.loc[data['tto_label']==2]
        x1 = 'oot_pcnt'
        x2 = 'oot_bad_rate'
    data_new = data[[var,target]].groupby([var]).count().reset_index()
    data_new['total'] = data_new[target].sum()
    data_new[x1] = data_new[target]/data_new['total']
    data_new[x1] = data_new[x1].map(lambda x: round(x * 100, 2))
    data_bad = data[[var,target]].groupby([var])[target].sum().reset_index()
    data_bad.columns = [var, 'bad']
    data_final = pd.merge(data_new, data_bad, on=var)
    data_final[x2] = data_final['bad'] / data_final[target]
    data_final[x2] = data_final[x2].map(lambda x: round(x * 100, 2))
    data_final = data_final[[var, x1, x2]]
    return data_final

def train_oot(data, target, tto_label, filename,outputh):
    
    total_columns = [i.lower() for i in data.columns]
    var_columns = [i for i in total_columns if (i.endswith('_label')) &(i!='tto_label')]
    data = data.loc[(data[target]==1)|(data[target]==0)]
    page = Page()
    for var in var_columns:
        data_var = data[['cid',target,var,tto_label]]
        train = cal_ratio2(data_var, var, 1, target)
        oot = cal_ratio2(data_var, var, 2, target)
        Line_1 = Line('{} 变量分BIN占比及bad_rate'.format(var), width=1200, height=500)

        Line_1.add("train_{}".format(var), train[var], train.iloc[:, 2], is_stack=False, is_datazoom_show=True,is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(194,53,49)"],
                  datazoom_range=[0, 100],legend_top='7%', is_toolbox_show=False, is_more_utils=False, legend_text_size="12")
        Line_1.add("oot_{}".format(var), oot[var], oot.iloc[:, 2], is_stack=False, is_datazoom_show=True,
                  datazoom_range=[0, 100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(47,69,84)"],
                  legend_top='7%', is_toolbox_show=False, is_more_utils=False, legend_text_size="12")


        Bar_1 = Bar('{} 变量分BIN占比及bad_rate'.format(var), width=1200, height=500)
        Bar_1.add("train_{}".format(var), train[var], train.iloc[:,1],is_stack=False, is_datazoom_show=True,datazoom_range=[0,100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(194,53,49)"],
                legend_top='7%',is_toolbox_show=False, is_more_utils=False,legend_text_size="12")
        Bar_1.add("oot_{}".format(var), oot[var], oot.iloc[:,1],is_stack=False, is_datazoom_show=True,datazoom_range=[0,100],is_splitline_show=False,yaxis_formatter="%",label_color=["rgb(47,69,84)"],
                legend_top='7%',is_toolbox_show=False, is_more_utils=False,legend_text_size="12")

        overlap1 = Overlap(width=1300, height=400)
        overlap1.add(Bar_1)
        overlap1.add(Line_1, yaxis_index=1, is_add_yaxis=True)
        page.add(overlap1)
    page.render(outputh + filename+ '_'+'train_oot_bin.html')


## eda函数
def Eda(data,outputh):
    df_date =data
    for row in df_date.columns:
        if (row != 'cid') & (row != 'add_day'):
            df__ = df_date[['add_day', 'cid', row]]
            df_1 = df__[(df__[row].notnull())].sort_values('add_day', ascending=True).reset_index(drop=True)
            df_1[row] = df_1[row].astype('float64')
            ### 计算总量
            df_2 = df__.groupby(['add_day']).count()
            df_tot = df_2.reset_index()
    
            df_tot.columns = ['add_day', 'cid', 'total_num']
            ###  计算缺失率
            df_3 = df__.groupby(['add_day']).count()
            df_missing = 1 - df_3.iloc[:, 1] / df_3.iloc[:, 0]
            df_missing = df_missing.to_frame().reset_index()
            df_missing.columns = ['add_day', 'missing_ratio']
            ## 计算总值
            df_tot_fin = df_tot[['add_day', 'total_num']]
    
            ### 计算均值、中位数、标准差、最大值、最小值(去掉缺失值之后)
            df_total = df_1.sort_index(ascending=False).groupby(['add_day'])[row].agg(
                {'mean', 'max', 'min', 'median', 'std'}).reset_index()
            ### 计算0值
            df_zero_1 = df_1.loc[df_1[row] != 0].groupby(['add_day']).count().reset_index()
            df_zero_1 = df_zero_1[['add_day', row]]
            df_zero_1.columns = ['add_day', 'new_' + row]
            df_zero_2 = df_1.groupby(['add_day']).count().reset_index()
            df_zero_fin = pd.merge(df_zero_1, df_zero_2, on='add_day')
            df_zero_fin['zero_ratio'] = 1- df_zero_fin['new_'+row] / df_zero_fin[row]
    
            ## 计算均值
            df_mean = df_total[['add_day', 'mean']]
            ## 计算标准差
            df_std = df_total[['add_day', 'std']]
            ## 计算中位数
            df_median = df_total[['add_day', 'median']]
            ## 计算最大值
            df_max = df_total[['add_day', 'max']]
            # 计算最小值
            df_min = df_total[['add_day', 'min']]
    
            fig = plt.figure(figsize=(20, 12))
            fig.subplots_adjust(left=0.3, bottom=0.2, right=0.7, top=0.8, hspace=0.5)
            date_format = mdate.DateFormatter("%Y-%m-%d")
    
            ax0 = plt.subplot(4, 2, 1)
            ax0.set_title('变量 {} 的 {} 趋势及分布'.format(row, '调用总量'))
            x0 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_tot_fin.add_day]
            ax0.bar(x0, df_tot_fin.total_num)
            # ax0.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax0.set_xticklabels([], rotation=45)
            ax0.set_ylim(ymin=0, ymax=1.2 * max(df_tot_fin.total_num))
            ax0.grid(True)
    
            ax1 = plt.subplot(4, 2, 2)
            ax1.set_title('变量 {} 的 {} 趋势及分布'.format(row, '缺失率'))
            x1 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_missing.add_day]
            ax1.plot(x1, df_missing.missing_ratio)
            ax1 = plt.gca()
            ax1.set_xticklabels([], rotation=45)
            # ax1.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax1.set_ylim(ymin=-0.2, ymax=1.2)
            ax1.grid(True)
            #
            ax2 = plt.subplot(4, 2, 3)
            ax2.set_title('变量 {} 的 {} 趋势及分布'.format(row, '均值'))
            x2 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_mean.add_day]
            ax2.plot(x2, df_mean['mean'])
            ax2 = plt.gca()
            ax2.set_xticklabels([], rotation=45)
            # ax2.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax2.set_ylim(ymin=0.9 * min(df_mean['mean']), ymax=1.2 * max(df_mean['mean']))
            ax2.grid(True)
            #
            ax3 = plt.subplot(4, 2, 4)
            ax3.set_title('变量 {} 的 {} 趋势及分布'.format(row, '中位值'))
            x3 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_median.add_day]
            ax3.plot(x3, df_median['median'])
            ax3 = plt.gca()
            ax3.set_xticklabels([], rotation=45)
            # ax3.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax3.set_ylim(ymin=0, ymax=1.2 * max(df_median['median']))
            ax3.grid(True)
            #
    
            ax4 = plt.subplot(4, 2, 5)
            ax4.set_title('变量 {} 的 {} 趋势及分布'.format(row, '0值'))
            x4 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_zero_fin.add_day]
            ax4.plot(x4, df_zero_fin['zero_ratio'])
            ax4 = plt.gca()
            ax4.set_xticklabels([], rotation=45)
            # ax4.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax4.set_ylim(ymin=-1, ymax=1)
            ax4.grid(True)
            #
            #
            ax5 = plt.subplot(4, 2, 6)
            ax5.set_title('变量 {} 的 {} 趋势及分布'.format(row, '标准差'))
            x5 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_std.add_day]
            ax5.plot(x5, df_std['std'])
            ax5 = plt.gca()
            ax5.set_xticklabels([], rotation=45)
            # ax5.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax5.set_ylim(ymin=0, ymax=1.5 * max(df_std['std']))
            ax5.grid(True)
            #
    
            ax6 = plt.subplot(4, 2, 7)
            ax6.set_title('变量 {} 的 {} 趋势及分布'.format(row, '最大值'))
            x6 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_max.add_day]
            ax6.plot(x6, df_max['max'])
            # ax6.set_xticklabels(x6, rotation=45)
    
            ax6 = plt.gca()
            # ax6.set_xticklabels(x6, rotation=45)
            ax6.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax6.set_ylim(ymin=0, ymax=1.2 * max(df_max['max']))
            ax6.grid(True)
    
            ax7 = plt.subplot(4, 2, 8)
            ax7.set_title('变量 {} 的 {} 趋势及分布'.format(row, '最小值'))
            x7 = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df_min.add_day]
            ax7.plot(x7, df_min['min'])
    
            # ax7.set_xticklabels(x7, rotation=45)
    
            ax7 = plt.gca()
            ax7.xaxis.set_major_formatter(date_format)
            # fig.autofmt_xdate(rotation=45)
            ax7.set_ylim(ymin=0, ymax=1.2 * max(df_min['min']))
            ax7.grid(True)
    
            plt.subplots_adjust(wspace=0.4, hspace=0.7)
            plt.tight_layout()
    
            plt.savefig(outputh  + row +'_eda'+ '.png')
            

def SplitData2(df, col, numOfSplit):
    '''
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数df=data
    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    '''
    df2 = df.copy()
    N = max(df2[col])-min(df2[col])
    n = N//numOfSplit
    splitPoint = [i*n for i in range(1,numOfSplit)]
    return splitPoint

    
def Chi2(df, total_col, bad_col,overallRate):
    '''
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :return: 卡方值
    '''
    df2 = df.copy()
    
    # 求出df中，总体的坏样本率和好样本率
    badRate = overallRate
    goodRate = 1 -overallRate
    df2['good'] = df2.apply(lambda x: x[total_col] -x[bad_col], axis=1)

    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x*badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    
    #存在分箱中有好客户为0或者坏客户为0的状况，故为0，则分母为0，详见function Replace
    badChi = [(i[0]-i[1])**2/Replace(i[0]) for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / Replace(i[0]) for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2  


def Replace(col):
    if col==0:
        return 1
    else :
        return col
    

##  卡方分箱（控制分箱数）'''
def ChiMerge(col,data,groupIntervals,max_interval,target,overallRate):
    df =data
    while (len(groupIntervals) > max_interval):  # 终止条件: 当前分箱数＝预设的分箱数
# 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
        chisqList = []
        for k in range(len(groupIntervals)):
            regroup =pd.DataFrame(np.zeros([1,2]),columns=['total','bad'])
    #        print(k)
            if k ==0:
                position_in_this_section =np.where(df[col]<=groupIntervals[k])[0]
            elif k==len(groupIntervals) -1:
                position_in_this_section =np.where(df[col]>groupIntervals[k])[0]            
            else:
                position_in_this_section =np.where((df[col]>groupIntervals[k-1])&(df[col]<=groupIntervals[k]))[0]      
            total = len(position_in_this_section)
            bad =sum(df[target].iloc[position_in_this_section])
            regroup =pd.DataFrame({'total':[total],'bad':[bad]})
            chisq = Chi2(regroup, 'total', 'bad', overallRate=overallRate)
            chisqList.append(chisq)
        best_comnbined = chisqList.index(min(chisqList))
        groupIntervals.remove(groupIntervals[best_comnbined])
    cutOffPoints =groupIntervals
    cutOffPoints.extend([-np.inf,np.inf])
    final_splitpoint =sorted(cutOffPoints)    
    return final_splitpoint

###
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''          七、auc ks lift swap分析自动化函数                '''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

## 创建样式
borders = xlwt.Borders()
borders.left = xlwt.Borders.THIN
borders.right = xlwt.Borders.THIN
borders.top = xlwt.Borders.THIN
borders.bottom = xlwt.Borders.THIN

style = xlwt.XFStyle() # 创建样式
font = xlwt.Font() 
alignment = xlwt.Alignment()
alignment.horz = 0x02 
alignment.vert = 0x01 
style.alignment = alignment
style.borders = borders
font.name = u'宋体'#字体为Arial
style.font.name = font.name
style.font.height=220#设置字体大小(220为11号字体，间隔40为一个字体)

style1 = xlwt.XFStyle()
alignment = xlwt.Alignment()
alignment.horz = 0x02 
alignment.vert = 0x01
style1.alignment = alignment
style1.borders = borders 
pattern = xlwt.Pattern()  # Create the pattern
pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
pattern.pattern_fore_colour = 5
style1.pattern = pattern
font = xlwt.Font() 
font.name = u'宋体'
style1.font.name = font.name
style1.font.height=220

style2 = xlwt.XFStyle() # 创建样式
font = xlwt.Font() 
alignment = xlwt.Alignment()
alignment.horz = 0x02 
alignment.vert = 0x01 
style2.alignment = alignment
style2.borders = borders
font.name = u'宋体'#字体为Arial
style2.font.name = font.name
style2.font.height=220#设置字体大小(220为11号字体，间隔40为一个字体)
style2.num_format_str='#0.00%'

style4 = xlwt.XFStyle()
alignment = xlwt.Alignment()
alignment.horz = 0x02 
alignment.vert = 0x01 
style4.alignment = alignment
style4.borders = borders
font = xlwt.Font() 
font.name = u'宋体'
pattern = xlwt.Pattern()  # Create the pattern
pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
pattern.pattern_fore_colour = 172
style4.pattern = pattern
font.bold = True
style4.font.bold = font.bold
style4.font.name = font.name
style4.font.height=220


## AucKsExcel样式
def AucKsExcelFormat(sheet,row_start_point,col_start_point,table_name,P1_name,P2_name,corr):
    
    sheet.write(row_start_point,col_start_point,table_name,style1)
    sheet.write(row_start_point+1,col_start_point,"AUC",style4)
    sheet.write(row_start_point+2,col_start_point,"K-S",style4)
    sheet.write(row_start_point+3,col_start_point,"相关性",style4)
    sheet.write(row_start_point,col_start_point+1,P1_name,style4)
    sheet.write(row_start_point,col_start_point+2,P2_name,style4)
    sheet.write_merge(row_start_point+3,row_start_point+3,col_start_point+1,col_start_point+2,corr,style)

## RankExcel样式
def RankExcelFormat(sheet,row_start_point,col_start_point,table_name,P1_name,P2_name,GroupNums=10):
    
    sheet.write(row_start_point,col_start_point,table_name,style1)
    sheet.write(row_start_point,col_start_point+1,P1_name,style4)
    sheet.write(row_start_point,col_start_point+2,P2_name,style4)
    for i in range(GroupNums):
        sheet.write(row_start_point+1+i,col_start_point,i+1,style4)
        
## SwapExcel样式
def SwapExcelFormat(sheet,row_start_point,col_start_point,table_name,P1_name,P2_name,GroupNums1,GroupNums2):
    
    col1 = sheet.col(0) 
    col1.width=200*20
    col2 = sheet.col(GroupNums2+3) 
    col2.width=200*20

    sheet.write(row_start_point,col_start_point,table_name,style1)
    sheet.write(row_start_point+1,col_start_point,P1_name,style4)
    sheet.write_merge(row_start_point,row_start_point,col_start_point+1,col_start_point+GroupNums2,P2_name,style4)
    for i in range(GroupNums1):
        sheet.write(row_start_point+2+i,col_start_point,i+1,style4)
    for i in range(GroupNums2):
        sheet.write(row_start_point+1,col_start_point+1+i,i+1,style4)
    sheet.write(row_start_point+GroupNums1+2,col_start_point,"总计",style4)
    sheet.write(row_start_point+1,col_start_point+GroupNums2+1,"总计",style4)

#    SwapExcelFormat(sheet1,18,0,"逾期率",P1_name,P2_name,GroupNums1,GroupNums2)
    
## Swap-in-out样式
def SwapInOutExcelFormat(sheet,row_start_point,col_start_point,P1_name,P2_name,CutNumP1,CutNumP2):
    sheet.write_merge(row_start_point,row_start_point+1,col_start_point,col_start_point,"swap",style1)
    merge_name ="{0}前{1}组与{2}前{3}组汇总分析".format(P1_name,CutNumP1,P2_name,CutNumP2)
    sheet.write_merge(row_start_point,row_start_point,col_start_point+1,col_start_point+6,merge_name,style4)    
    col_label =['label','tot','bad','per_rate','bad_rate','bad_rate_acu']
    for i in range(len(col_label)):
        col =col_label[i]
        sheet.write(row_start_point+1,col_start_point+1+i,col,style4)
    for i in range(1,5):
        sheet.write(row_start_point+1+i,col_start_point,i,style4)


 
## 等分十段，给每个人打上打上分组标签
def RankLabel(Pvalue,Numcutoff=10):   
    
    Pvalue_unique =np.unique(Pvalue)
    splitpoint =list(pd.qcut(Pvalue_unique, Numcutoff, retbins=True)[-1])
    Plabel =Pvalue.copy()
    eps =1e-06
    for i in range(1,Numcutoff+1):
        if i==1:
            position_in_this_section =np.where(Pvalue<splitpoint[i]+eps)[0]
        else:
            position_in_this_section =np.where((Pvalue>splitpoint[i-1])&(Pvalue<splitpoint[i]+eps))[0]
        Plabel.iloc[position_in_this_section] =i
    Plabel =Plabel.astype(int)
        
    return splitpoint, Plabel
   

## swap分析
def SwapData(df, y, P1, P2,GroupNums1,GroupNums2):
#    df =df_swap.copy()
    P1_splitpoint,P1_label_tmp =RankLabel(df[P1],Numcutoff =GroupNums1)
    P2_splitpoint,P2_label_tmp =RankLabel(df[P2],Numcutoff =GroupNums2)
    
    df_tmp =pd.concat([df.loc[:,[P1,P2,y]],pd.DataFrame(P1_label_tmp.values,columns=['P1_label']),pd.DataFrame(P2_label_tmp.values,columns=['P2_label'])],axis=1)
    
    P1_num_bad_count =pd.pivot_table(df_tmp, index=['P1_label'], values=y,aggfunc=[len,np.sum])
    P2_num_bad_count =pd.pivot_table(df_tmp, index=['P2_label'], values=y,aggfunc=[len,np.sum])

    df_swap_len_tmp =pd.pivot_table(df_tmp, index=['P1_label'], columns=['P2_label'], values=y,aggfunc=[len])
    
    ## 增加汇总行和列
    df_swap_len_tmp =pd.DataFrame(df_swap_len_tmp.values,index=range(len(df_swap_len_tmp)),columns=range(df_swap_len_tmp.shape[1]))
    tmp_col =pd.DataFrame(df_swap_len_tmp.sum(axis=1),columns=[df_swap_len_tmp.shape[1]])
    df_swap_len =df_swap_len_tmp.join(tmp_col)
    tmp_row =pd.DataFrame(df_swap_len_tmp.sum(axis=0),columns=[df_swap_len_tmp.shape[0]]).T
    df_swap_len =df_swap_len.append(tmp_row)
    df_swap_len.iloc[df_swap_len.shape[0]-1,df_swap_len.shape[1]-1] =len(df)
                
    df_swap_sum_tmp =pd.pivot_table(df_tmp, index=['P1_label'], columns=['P2_label'], values=y,aggfunc=[np.sum]) 
    ## 增加汇总行和列
    df_swap_sum_tmp =pd.DataFrame(df_swap_sum_tmp.values,index=range(len(df_swap_sum_tmp)),columns=range(df_swap_len_tmp.shape[1]))
    tmp_col =pd.DataFrame(df_swap_sum_tmp.sum(axis=1),columns=[df_swap_sum_tmp.shape[1]])
    df_swap_sum =df_swap_sum_tmp.join(tmp_col)
    tmp_row =pd.DataFrame(df_swap_sum_tmp.sum(axis=0),columns=[df_swap_sum_tmp.shape[0]]).T
    df_swap_sum =df_swap_sum.append(tmp_row)
    df_swap_sum.iloc[df_swap_sum.shape[0]-1,df_swap_sum.shape[1]-1] =sum(df[y])
       
    ##为缺失分组添加一个样本：
    df_swap_len =df_swap_len.fillna(1)
    df_swap_sum =df_swap_sum.fillna(0)

    df_swap_len_rate =pd.DataFrame(df_swap_len.values/len(df))   
     
    df_swap_sum_rate =pd.DataFrame(df_swap_sum.values/df_swap_len.values)
       
    return P1_num_bad_count,P2_num_bad_count,df_swap_len,df_swap_sum,df_swap_len_rate,df_swap_sum_rate

## swapinout data
def SwapInOutData(df_swap_len,df_swap_sum,CutNumP1,CutNumP2,GroupNums1,GroupNums2):
    
    df_tmp =pd.DataFrame(['in_in','swap_in','swap_out','out_out'])    
    len_in_in_data =np.array(df_swap_len.iloc[:CutNumP1,:CutNumP2]).sum()
    len_swap_in_data =np.array(df_swap_len.iloc[:CutNumP1,CutNumP2:GroupNums2]).sum()
    len_swap_out_data =np.array(df_swap_len.iloc[CutNumP1:GroupNums1,:CutNumP2]).sum()
    len_out_out_data =np.array(df_swap_len.iloc[CutNumP1:GroupNums1,CutNumP2:GroupNums2]).sum()
    
    sum_in_in_data =np.array(df_swap_sum.iloc[:CutNumP1,:CutNumP2]).sum()
    sum_swap_in_data =np.array(df_swap_sum.iloc[:CutNumP1,CutNumP2:GroupNums2]).sum()
    sum_swap_out_data =np.array(df_swap_sum.iloc[CutNumP1:GroupNums1,:CutNumP2]).sum()
    sum_out_out_data =np.array(df_swap_sum.iloc[CutNumP1:GroupNums1,CutNumP2:GroupNums2]).sum()
    
    tot =pd.Series([len_in_in_data,len_swap_in_data,len_swap_out_data,len_out_out_data])
    bad =pd.Series([sum_in_in_data,sum_swap_in_data,sum_swap_out_data,sum_out_out_data])
    bad_cnt =pd.Series([bad[:1].sum(),bad[:2].sum(),bad[:3].sum(),bad[:4].sum()])
    
    per_rate =tot/sum(tot)
    bad_rate =bad/tot
    bad_rate_acu =bad_cnt/sum(tot)
    
    df_tot_bad =pd.concat([df_tmp,tot,bad],axis=1)
    df_tot_bad.columns =range(df_tot_bad.shape[1])

    df_tot_bad_rate =pd.concat([per_rate,bad_rate,bad_rate_acu],axis=1)
    
    return df_tot_bad,df_tot_bad_rate

def WriteData(df,sheet,row_start_point,col_start_point):

    for col in range(df.shape[1]):
        tmp =df[df.columns[col]]
        for row in range(df.shape[0]):
            try:
                sheet.write(row_start_point+row,col_start_point+col,tmp[row].astype(float),style)
            except:
                sheet.write(row_start_point+row,col_start_point+col,tmp[row],style)

def WriteData2(df,sheet,row_start_point,col_start_point,GroupNums1,GroupNums2):
    """
    df =Rank_bad.copy()
    sheet =sheet1
    row_start_point =2
    col_start_point =5
    """  
    for row in range(GroupNums1):
        tmp =df.iloc[:,0]
        try:
            sheet.write(row_start_point+row,col_start_point,tmp[row].astype(float),style2)
        except:
            sheet.write(row_start_point+row,col_start_point,tmp[row],style2)

    for row in range(GroupNums2):
        tmp =df.iloc[:,1]
        try:
            sheet.write(row_start_point+row,col_start_point+1,tmp[row].astype(float),style2)
        except:
            sheet.write(row_start_point+row,col_start_point+1,tmp[row],style2)
    
def WriteData3(df,sheet,row_start_point,col_start_point):

    for col in range(df.shape[1]):
        tmp =df[df.columns[col]]
        for row in range(df.shape[0]):
            try:
                sheet.write(row_start_point+row,col_start_point+col,tmp[row].astype(float),style2)
            except:
                sheet.write(row_start_point+row,col_start_point+col,tmp[row],style2)
  

                
def Swap(df,y,P1,P2,P1_name,P2_name,CutNumP1,CutNumP2,filename,outputh,GroupNums1=10,GroupNums2=10):
    
    """
    df =df_swap.copy()
    filename ="test3.xls"
    GroupNums1 =10
    GroupNums2 =3
    """
    ks_value,bad_percent,good_percent=cal_ks(np.array(df[P1]),np.array(df[y]),section_num=10)
    false_positive_rate, recall, thresholds = roc_curve(df[y], df[P1])
    roc_auc=auc(false_positive_rate,recall)
    P1_ks =np.max(ks_value)
    P1_auc =roc_auc
    
    ks_value,bad_percent,good_percent=cal_ks(np.array(df[P2]),np.array(df[y]),section_num=10)
    false_positive_rate, recall, thresholds = roc_curve(df[y], df[P2])
    roc_auc=auc(false_positive_rate,recall)
    P2_ks =np.max(ks_value)
    P2_auc =roc_auc

    P1_rank,P2_rank,df_swap_len,df_swap_sum,df_swap_len_rate,df_swap_sum_rate =SwapData(df, y,P1,P2,GroupNums1,GroupNums2)
    
    ##计算分组坏账率
    P1_rank_bad =P1_rank.iloc[:,1]/P1_rank.iloc[:,0]
    P2_rank_bad =P2_rank.iloc[:,1]/P2_rank.iloc[:,0]
    Rank_bad =pd.concat([pd.DataFrame(P1_rank_bad.values,columns=['P1_rank_bad']),pd.DataFrame(P2_rank_bad.values,columns=['P2_rank_bad'])],axis=1)
     
    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet("交叉分析")
    
    ##auc ks rank写入
    corr =np.corrcoef(df[P1],df[P2])[0,1]
    AucKsExcelFormat(sheet1,1,0,"模型效果",P1_name,P2_name,corr.round(4))
    sheet1.write(2,1,P1_auc.round(4),style)
    sheet1.write(2,2,P2_auc.round(4),style)
    sheet1.write(3,1,P1_ks.round(4),style)
    sheet1.write(3,2,P2_ks.round(4),style)
    
    ## Rank写入
    RankExcelFormat(sheet1,1,4,"评分排序",P1_name,P2_name,GroupNums=10)
    WriteData2(Rank_bad,sheet1,2,5,GroupNums1,GroupNums2)
    
    ## Swap写入
    SwapExcelFormat(sheet1,18,0,"逾期率",P1_name,P2_name,GroupNums1,GroupNums2)
    WriteData3(df_swap_sum_rate,sheet1,20,1)
    
    SwapExcelFormat(sheet1,18,GroupNums2+3,"人数占比",P1_name,P2_name,GroupNums1,GroupNums2)
    WriteData3(df_swap_len_rate,sheet1,20,GroupNums2+4)
    
    SwapExcelFormat(sheet1,30+GroupNums1,0,"总体人数分布",P1_name,P2_name,GroupNums1,GroupNums2)
    WriteData(df_swap_len,sheet1,32+GroupNums1,1)
    
    SwapExcelFormat(sheet1,30+GroupNums1,GroupNums2+3,"逾期人数分布",P1_name,P2_name,GroupNums1,GroupNums2)
    WriteData(df_swap_sum,sheet1,32+GroupNums1,GroupNums2+4)

    ## swapinout data 写入
    df_tot_bad,df_tot_bad_rate =SwapInOutData(df_swap_len,df_swap_sum,CutNumP1,CutNumP2,GroupNums1,GroupNums2)
    SwapInOutExcelFormat(sheet1,23+GroupNums1,0,P1_name,P2_name,CutNumP1,CutNumP2)
    WriteData(df_tot_bad,sheet1,25+GroupNums1,1)
    WriteData3(df_tot_bad_rate,sheet1,25+GroupNums1,4)

    ## 文件输出
    filename =filename
    outputh = outputh
    book.save(os.path.join(outputh, filename))












    

