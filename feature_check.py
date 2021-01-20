
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm
import lightgbm as lgb

# 不限制最大显示列数
pd.set_option('display.max_columns', None)
# 不限制最大显示行数
pd.set_option('display.max_rows', None)

path = '/home/bb5/xw/mxg_scores/feature_check'
offline =pd.read_csv(path+'/20201222_pg_offline.csv')
online = pd.read_csv(path+'/20201222_pg_online.csv')
scores = pd.read_csv(path+'/20201222_pg_scores.csv')

# 缺失率
print('offline',offline.isna().sum()/len(offline))
print('online',online.isna().sum()/len(online))



offline.fillna(-99999,inplace=True)
online.fillna(-99999,inplace=True)

df = pd.merge(offline,online,left_on='biz_id',right_on='biz_id', suffixes=('_offline', '_online'))

# 读取保存好的 model 和feature names
path ='/home/bb5/xw/mxg_scores'
input_xgb_features_path=path+'/lgbm_v1_features.csv'
chosen_feature_pd=pd.read_csv(input_xgb_features_path,encoding='UTF-8')
chosen_feature=list(np.array(chosen_feature_pd.iloc[:,0],dtype='str'))
print('chosen_feature: ' ,chosen_feature)

input_model_path=path+'/lgbm_v1.model'
lgbm_model = lgb.Booster(model_file=input_model_path)


count = 0
f_list = []
for i in chosen_feature:
    count += 1
    f_offline = str(i) + '_offline'
    f_online = str(i) + '_online'
    tmp = df[df[f_offline].round(5) == df[f_online].round(5)]
    acc = round(len(tmp) / len(df), 2)
    print(count, '  acc  ', i, ' ', acc)
    if acc <= 0.99:
        f_list.append(i)



# 打印变量
for i in f_list:
    f_offline = str(i) + '_offline'
    f_online = str(i) + '_online'
    a = df[df[f_offline] != df[f_online]].loc[:, ['biz_id','cust_id', f_offline, f_online]]
    print(a.head(10))
    print('\t')
    print('\t')
    print('\t')



# 打印份数


path = '/home/bb5/xw/mxg_scores/feature_check'
offline =pd.read_csv(path+'/20201221_pg_offline.csv')
online = pd.read_csv(path+'/20201221_pg_online.csv')
scores = pd.read_csv(path+'/20201221_pg_scores.csv')


factor=40/(np.log(120/40)-np.log(60/40))
offset=500-factor*np.log(60/40)
df =pd.merge(online,scores,left_on='biz_id',right_on='biz_id')
df['preds'] = lgbm_model.predict(df.loc[:, chosen_feature])
df['score'] = (np.log((1 - df['preds'])/df['preds']) * factor + offset).round(0)

tmp1 = df[df['preds']!=df['mx_firstloan_probability1']]
print('模型概率不相同',len(tmp1)/len(df))

tmp2 = df[df['score']!=df['riskscore']]
tmp2= tmp2[['biz_id','score','riskscore','preds','mx_firstloan_probability1']]
print('模型分数不相同',len(tmp2)/len(df))



# 模型重新验证
from  toad.metrics import KS,KS_bucket,AUC,PSI

sample = pd.read_csv(path+'/20201221_pg.csv')
sample['y']=sample['overdue_days'].map(lambda x: 1 if x>=7 else 0)

sample01=sample[sample['apply_time']<='2020-11-15']
sample02=sample[sample['apply_time']>='2020-11-15']


X,Y = sample01.loc[:, chosen_feature],sample01['y']
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# 读取model文件，KS值和AUC值分别为[0.334756960344333, 0.7248087864585016]


X,Y = sample02.loc[:, chosen_feature],sample02['y']
preds = lgbm_model.predict(X.loc[:, chosen_feature])
ks = KS(preds,Y)
auc = AUC(preds,Y)
print('读取model文件，KS值和AUC值分别为{0}'.format([ks, auc]))
# 读取model文件，KS值和AUC值分别为[0.32038723782159423, 0.7152151959711868]
