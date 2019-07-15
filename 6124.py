import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kurtosis
import time
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
path = 'D:/jupyter/data/paipai/data/'

user_repay_logs = pd.read_csv(path+'user_repay_logs.csv', parse_dates=['due_date', 'repay_date'])
test = pd.read_csv(path+'test.csv',parse_dates=['auditing_date', 'due_date'])
train = pd.read_csv(path+'train.csv', parse_dates=['due_date', 'repay_date'])
repay_log_df = user_repay_logs.loc[user_repay_logs['order_id'] ==1]
del repay_log_df['order_id']
listing_info = pd.read_csv(path+'listing_info.csv',parse_dates=['auditing_ date'])
repay_log_df = repay_log_df.merge(listing_info[['listing_id','auditing_date']],on = 'listing_id',how = 'left')
del user_repay_logs

repay_log_df= pd.concat((train,repay_log_df))
repay_log_df = repay_log_df.drop_duplicates('listing_id').reset_index(drop=True)
repay_log_df['repay_date'] = pd.to_datetime(repay_log_df['repay_date'])
repay_log_df['repay_date'] = repay_log_df['repay_date'].replace("\\N",'2200-01-01')
repay_log_df.loc[repay_log_df['repay_date']=='2200-01-01','repay_amt'] = 0
repay_log_df['auditing_date'] = pd.to_datetime(repay_log_df['auditing_date'])

repay_log_df = repay_log_df.drop(repay_log_df[(repay_log_df['due_date'] - repay_log_df['auditing_date']).dt.days == 29].index).reset_index(drop=True)
repay_log_df = repay_log_df.drop(repay_log_df[(repay_log_df['due_date'] - repay_log_df['auditing_date']).dt.days > 31].index).reset_index(drop=True)
repay_log_df = repay_log_df.drop(repay_log_df[(repay_log_df['due_date'] - repay_log_df['auditing_date']).dt.days < 28].index).reset_index(drop=True)

train_1 = repay_log_df.loc[repay_log_df['auditing_date']>= '2019-1-01'].reset_index(drop=True)
repay_log_df_1 = repay_log_df[~repay_log_df['listing_id'].isin(train_1['listing_id'].unique())]
train_2 = repay_log_df_1.loc[repay_log_df['auditing_date']>= '2018-11-01'].reset_index(drop=True)
repay_log_df_2 = repay_log_df_1[~repay_log_df_1['listing_id'].isin(train_2['listing_id'].unique())]
train_3 = repay_log_df_2.loc[repay_log_df['auditing_date']>= '2018-09-01'].reset_index(drop=True)
repay_log_df_3 = repay_log_df_2[~repay_log_df_2['listing_id'].isin(train_3['listing_id'].unique())]
train_4 = repay_log_df_3.loc[repay_log_df['auditing_date']>= '2018-07-01'].reset_index(drop=True)
repay_log_df_4 = repay_log_df_3[~repay_log_df_3['listing_id'].isin(train_4['listing_id'].unique())]
train_5 = repay_log_df_4.loc[repay_log_df['auditing_date']>= '2018-05-01'].reset_index(drop=True)
repay_log_df_5 = repay_log_df_4[~repay_log_df_4['listing_id'].isin(train_5['listing_id'].unique())]
train_6 = repay_log_df_5.loc[repay_log_df['auditing_date']>= '2018-03-01'].reset_index(drop=True)
repay_log_df_6 = repay_log_df_5[~repay_log_df_5['listing_id'].isin(train_6['listing_id'].unique())]
# train_7 = repay_log_df_6.loc[repay_log_df['auditing_date']>= '2018-01-01'].reset_index(drop=True)
# repay_log_df_7 = repay_log_df_6[~repay_log_df_6['listing_id'].isin(train_7['listing_id'].unique())]

train_df = pd.concat([train_1,train_2,train_3,train_4,train_5,train_6])#,train_7
df = pd.concat([train_df, test], axis=0, ignore_index=True)

listing_info_df = pd.read_csv(path+'listing_info.csv')
listing_info_df['term_rate']=listing_info_df['term'].map(lambda x:str(x))+str('+')+listing_info_df['rate'].map(lambda x:str(x))
del listing_info_df['user_id'], listing_info_df['auditing_date']
df = df.merge(listing_info_df, on='listing_id', how='left')

le =LabelEncoder()
df['term_rate'] = le.fit_transform(df['term_rate'])
del listing_info_df


user_behavior_logs = pd.read_csv(path+'user_behavior_logs.csv',parse_dates=['behavior_time'])
behavior = pd.get_dummies(user_behavior_logs, columns=['behavior_type'])
user_be = behavior.drop('behavior_time',axis=1)
user_behavior_sum = user_be.groupby(['user_id']).sum().reset_index()
df = df.merge(user_behavior_sum, on='user_id', how='left')

user_behavior_logs['behavior_time'] = pd.to_datetime(user_behavior_logs['behavior_time'])
user_behavior_logs['hours'] = user_behavior_logs['behavior_time'].dt.hour
user_behavior_logs['time_category'] = 0
user_behavior_logs.loc[(user_behavior_logs['hours']>1)&(user_behavior_logs['hours']<=5),'time_category']=1
user_behavior_logs.loc[(user_behavior_logs['hours']>5)&(user_behavior_logs['hours']<=11),'time_category']=2
user_behavior_logs.loc[(user_behavior_logs['hours']>11)&(user_behavior_logs['hours']<=14),'time_category']=3
user_behavior_logs.loc[(user_behavior_logs['hours']>14)&(user_behavior_logs['hours']<=18),'time_category']=4
user_behavior_logs.loc[(user_behavior_logs['hours']>18)&(user_behavior_logs['hours']<=21),'time_category']=5
user_behavior_logs = pd.get_dummies(user_behavior_logs, columns=['time_category'])

user_be = user_behavior_logs.drop(['hours','behavior_type','behavior_time'],axis=1)
user_be = user_be.drop_duplicates('user_id').reset_index(drop=True)
del user_behavior_logs
df = df.merge(user_be, on='user_id', how='left')


# 表中有少数user不止一条记录，因此按日期排序，去重，只保留最新的一条记录。
user_info_df = pd.read_csv(path+'user_info.csv', parse_dates=['reg_mon', 'insertdate'])
user_info_df.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)
user_info_df = user_info_df.sort_values(by='info_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

user_info_df['info_insert_date_year']= user_info_df['info_insert_date'].dt.year
user_info_df['info_insert_date_month']= user_info_df['info_insert_date'].dt.month
user_info_df['info_insert_date_dayofweek']= user_info_df['info_insert_date'].dt.dayofweek
user_info_df['reg_mon_date_year']= user_info_df['reg_mon'].dt.year
user_info_df['reg_mon_date_month']= user_info_df['reg_mon'].dt.month
df = df.merge(user_info_df, on='user_id', how='left')
df['Whether_work_outside'] =  0
df.loc[df['cell_province'] == df['id_province'],'Whether_work_outside']=1

# 同上
user_tag_df = pd.read_csv(path+'user_taglist.csv', parse_dates=['insertdate'])
user_tag_df.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
user_tag_df['taglist_len'] = user_tag_df['taglist'].apply(lambda x : len(x.split('|')))
user_tag_df = user_tag_df.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)

user_tag_df['tag_insert_date_year']= user_tag_df['tag_insert_date'].dt.year
user_tag_df['tag_insert_date_month']= user_tag_df['tag_insert_date'].dt.month
user_tag_df['tag_insert_date_dayofweek']= user_tag_df['tag_insert_date'].dt.dayofweek

df = df.merge(user_tag_df, on='user_id', how='left')
del user_tag_df

tag_list = pd.read_csv('taglist_30.csv')
df = pd.merge(df,tag_list,on='user_id',how='left')
del tag_list

list_feature = pd.read_csv('listing_feature.csv')
df = pd.merge(df,list_feature[['diff_days','listing_id','1_repay_day']],on='listing_id',how='left')
del list_feature

df['early_repay_days'] = (df['due_date'] - df['repay_date']).dt.days
df['early_repay_days'] = df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
df['late_repay_days'] = (df['repay_date'] - df['auditing_date']).dt.days
df['late_repay_days'] = df['late_repay_days'].apply(lambda x: x if x <= 31 else 32)

date_cols = ['auditing_date', 'due_date']#, 'reg_mon', 'info_insert_date', 'tag_insert_date'
for f in date_cols:
    if f in ['auditing_date', 'due_date']:
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek
df.drop(columns=['due_date', 'reg_mon', 'tag_insert_date'], axis=1, inplace=True)

temp_list = ['id_city','term_rate','auditing_date_dayofweek','auditing_date_day','age','cell_province']
agg_fun = ['mean','std']
while temp_list:
    s = temp_list.pop()
    for i in ['early_repay_days','diff_days']:
        temp = df[[s, i]].groupby(s).agg(agg_fun)
        temp.columns = [s + '_' + i + '_' + k for k in agg_fun]
        df = df.join(temp, on=s)

cate_cols = ['gender', 'cell_province', 'id_province','id_city']#,
for s in cate_cols:
    temp = df.groupby(s).size().rename(s + '_' + 'encode')
    df = df.join(temp, on=s)
df.drop(columns=cate_cols, axis=1, inplace=True)

train_1 = df[df['listing_id'].isin(train_1['listing_id'].unique())]
train_2 = df[df['listing_id'].isin(train_2['listing_id'].unique())]
train_3 = df[df['listing_id'].isin(train_3['listing_id'].unique())]
train_4 = df[df['listing_id'].isin(train_4['listing_id'].unique())]
train_5 = df[df['listing_id'].isin(train_5['listing_id'].unique())]
train_6 = df[df['listing_id'].isin(train_6['listing_id'].unique())]
# train_7 = df[df['listing_id'].isin(train_7['listing_id'].unique())]
test = df[df['listing_id'].isin(test['listing_id'].unique())]

repay_log_df['repay'] = repay_log_df['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
repay_log_df['early_repay_days'] = (repay_log_df['repay_date'] - repay_log_df['auditing_date']).dt.days
repay_log_df['early_repay_days'] = repay_log_df['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
repay_log_df['late_repay_days'] = (repay_log_df['repay_date'] - repay_log_df['auditing_date']).dt.days
repay_log_df['late_repay_days'] = repay_log_df['late_repay_days'].apply(lambda x: x if x <= 31 else 32)

repay_log_df = repay_log_df.sort_values(by = ['user_id','auditing_date'])
repay_log_df['count'] = repay_log_df.groupby(['user_id'])['auditing_date'].rank(method='first')
repay_log_df['diff_days'] = repay_log_df['auditing_date'].diff().dt.days
repay_log_df.loc[repay_log_df['count'] == 1.0,'diff_days'] = None
del repay_log_df['count']

repay_log_df_1 = repay_log_df[~repay_log_df['listing_id'].isin(train_1['listing_id'].unique())]
repay_log_df_2 = repay_log_df_1[~repay_log_df_1['listing_id'].isin(train_2['listing_id'].unique())]
repay_log_df_3 = repay_log_df_2[~repay_log_df_2['listing_id'].isin(train_3['listing_id'].unique())]
repay_log_df_4 = repay_log_df_3[~repay_log_df_3['listing_id'].isin(train_4['listing_id'].unique())]
repay_log_df_5 = repay_log_df_4[~repay_log_df_4['listing_id'].isin(train_5['listing_id'].unique())]
repay_log_df_6 = repay_log_df_5[~repay_log_df_5['listing_id'].isin(train_6['listing_id'].unique())]
# repay_log_df_7 = repay_log_df_6[~repay_log_df_6['listing_id'].isin(train_7['listing_id'].unique())]

for f in ['listing_id', 'due_date', 'repay_date', 'repay_amt']:#
    del repay_log_df[f]
    del repay_log_df_1[f]
    del repay_log_df_2[f]
    del repay_log_df_3[f]
    del repay_log_df_4[f]
    del repay_log_df_5[f]
    del repay_log_df_6[f]
#     del repay_log_df_7[f]

group = repay_log_df.groupby('user_id', as_index=False)
group_1 = repay_log_df_1.groupby('user_id', as_index=False)
group_2 = repay_log_df_2.groupby('user_id', as_index=False)
group_3 = repay_log_df_3.groupby('user_id', as_index=False)
group_4 = repay_log_df_4.groupby('user_id', as_index=False)
group_5 = repay_log_df_5.groupby('user_id', as_index=False)
group_6 = repay_log_df_6.groupby('user_id', as_index=False)
# group_7 = repay_log_df_7.groupby('user_id', as_index=False)

repay_log_df = repay_log_df.merge(
    group['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_1 = repay_log_df_1.merge(
    group_1['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_2 = repay_log_df_2.merge(
    group_2['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_3 = repay_log_df_3.merge(
    group_3['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_4 = repay_log_df_4.merge(
    group_4['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_5 = repay_log_df_5.merge(
    group_5['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
repay_log_df_6 = repay_log_df_6.merge(
    group_6['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')
# repay_log_df_7 = repay_log_df_7.merge(
#     group_7['repay'].agg({'repay_mean': 'mean','repay_std': 'std'}), on='user_id', how='left')

repay_log_df = repay_log_df.merge(
    group['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_1 = repay_log_df_1.merge(
    group_1['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_2 = repay_log_df_2.merge(
    group_2['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_3 = repay_log_df_3.merge(
    group_3['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_4 = repay_log_df_4.merge(
    group_4['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_5 = repay_log_df_5.merge(
    group_5['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
repay_log_df_6 = repay_log_df_6.merge(
    group_6['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
# repay_log_df_7 = repay_log_df_7.merge(
#     group_7['early_repay_days'].agg({
#         'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
#         'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
#     }), on='user_id', how='left')

for f in ['repay','auditing_date','due_amt', 'early_repay_days', 'late_repay_days','diff_days']:#
    del repay_log_df[f]
    del repay_log_df_1[f]
    del repay_log_df_2[f]
    del repay_log_df_3[f]
    del repay_log_df_4[f]
    del repay_log_df_5[f]
    del repay_log_df_6[f]
#     del repay_log_df_7[f]

repay_log_df = repay_log_df.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_1 = repay_log_df_1.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_2 = repay_log_df_2.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_3 = repay_log_df_3.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_4 = repay_log_df_4.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_5 = repay_log_df_5.drop_duplicates('user_id').reset_index(drop=True)
repay_log_df_6 = repay_log_df_6.drop_duplicates('user_id').reset_index(drop=True)
# repay_log_df_7 = repay_log_df_7.drop_duplicates('user_id').reset_index(drop=True)

train_1 = train_1.merge(repay_log_df_1, on='user_id', how='left')
train_2 = train_2.merge(repay_log_df_2, on='user_id', how='left')
train_3 = train_3.merge(repay_log_df_3, on='user_id', how='left')
train_4 = train_4.merge(repay_log_df_4, on='user_id', how='left')
train_5 = train_5.merge(repay_log_df_5, on='user_id', how='left')
train_6 = train_6.merge(repay_log_df_6, on='user_id', how='left')
# train_7 = train_7.merge(repay_log_df_7, on='user_id', how='left')
test  = test.merge(repay_log_df, on='user_id', how='left')

train = pd.concat([train_1,train_2,train_3,train_4,train_5,train_6])#,train_7
# train = train[~train['user_id'].isin(test['user_id'].unique())]

train_df = train.copy()
test_df = test.copy()
train_df = train_df.loc[train_df['auditing_date']<'2019-01-01'].reset_index(drop=True)
train_df.loc[train_df['early_repay_days']<0,'early_repay_days'] = 32
train_num = train_df.shape[0]

clf_labels =  train_df['early_repay_days'].values
amt_labels = train_df['repay_amt'].values
train_due_amt_df = train_df[['due_amt']]
df1 = pd.concat([train_df,test_df])
drop_list= ['age','info_insert_date','taglist','repay_amt','repay_date','early_repay_days','late_repay_days','auditing_date','listing_id','user_id']
df1 = df1.drop(columns=drop_list, axis=1)
train_values, test_values =df1[:train_num], df1[train_num:]

import gc
gc.collect()
import os
import multiprocessing

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 指定使用id=2的gpu
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2222)
params = {
    'application': 'multiclass',
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'random_state': 2019,
    'n_jobs': multiprocessing.cpu_count()
    #         'device' : "gpu",
    #         'gpu_platform_id' : 0,
    #         'gpu_device_id' : 0,
}

clf = lgb.LGBMClassifier(**params)

amt_oof = np.zeros(train_num)
prob_oof = np.zeros((train_num, 33))
test_pred_prob = np.zeros((test.shape[0], 33))
for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, clf_labels)):
    print(str(i + 1), 'fold...')
    t = time.time()

    trn_x, trn_y = train_values.loc[trn_idx], clf_labels[trn_idx]
    val_x, val_y = train_values.loc[val_idx], clf_labels[val_idx]
    val_repay_amt = amt_labels[val_idx]
    val_due_amt = train_due_amt_df.iloc[val_idx]
    val_weight = (train_values['due_amt'] / train_values['due_amt'].mean()).iloc[val_idx],
    #     trn_weight = (train_values['due_amt']/train_values['due_amt'].mean()).iloc[trn_idx]
    clf.fit(
        trn_x, trn_y,
        eval_sample_weight=list(val_weight),
        #         sample_weight = list(trn_weight),
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        early_stopping_rounds=50, verbose=20,
    )
    # shepe = (-1, 33)
    val_pred_prob_everyday = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    prob_oof[val_idx] = val_pred_prob_everyday
    val_pred_prob_today = [val_pred_prob_everyday[i][int(val_y[i])] for i in range(val_pred_prob_everyday.shape[0])]

    val_pred_repay_amt = val_due_amt['due_amt'].values * val_pred_prob_today
    print('val rmse:', np.sqrt(mean_squared_error(val_repay_amt, val_pred_repay_amt)))
    print('val mae:', mean_absolute_error(val_repay_amt, val_pred_repay_amt))
    amt_oof[val_idx] = val_pred_repay_amt
    test_pred_prob += clf.predict_proba(test_values.values, num_iteration=clf.best_iteration_) / skf.n_splits
    print('runtime: {}\n'.format(time.time() - t))
print('\ncv rmse:', np.sqrt(mean_squared_error(amt_labels, amt_oof)))
print('cv mae:', mean_absolute_error(amt_labels, amt_oof))
print('cv logloss:', log_loss(clf_labels, prob_oof))
print('cv acc:', accuracy_score(clf_labels, np.argmax(prob_oof, axis=1)))

"""导出根据结果sub"""
test_df = pd.read_csv(path + 'test.csv', parse_dates=['auditing_date', 'due_date'])
sub = test_df[['listing_id', 'auditing_date', 'due_amt', 'due_date']]
sub['due_days'] = (sub['due_date'] - sub['auditing_date']).dt.days
prob_cols = ['prob_{}'.format(i) for i in range(33)]
for i, f in enumerate(prob_cols):
    sub[f] = test_pred_prob[:, i]
cols_28 = ['prob_{}'.format(i) for i in range(29)]
cols_30 = ['prob_{}'.format(i) for i in range(31)]
su_28 = sub.loc[sub['due_days'] == 28]
for i, f in enumerate(cols_28):
    #     su_28[f] = su_28[f]+(i)*(su_28['prob_31']+su_28['prob_30']+su_28['prob_29'])/sum(list(range(29)))
    #     su_28[f] = su_28[f]*(1+(su_28['prob_31']+su_28['prob_30']+su_28['prob_29']))
    su_28[f] = su_28[f] + (su_28['prob_31'] + su_28['prob_30'] + su_28['prob_29']) * su_28[f] / su_28[cols_28].sum(
        axis=1)
su_30 = sub.loc[sub['due_days'] == 30]
for i, f in enumerate(cols_30):
    #     su_30[f] = su_30[f]+(i)*(su_30['prob_31'])/sum(list(range(31)))
    #      su_30[f] = su_30[f]*(1+su_30['prob_31'])
    su_30[f] = su_30[f] + su_30['prob_31'] * su_30[f] / su_30[cols_30].sum(axis=1)
sub = sub[~sub['listing_id'].isin(su_28['listing_id'].unique())]
sub = sub[~sub['listing_id'].isin(su_30['listing_id'].unique())]
sub = pd.concat((sub, su_28, su_30))
sub_example = pd.read_csv(path + 'submission.csv', parse_dates=['repay_date'])
sub_example = sub_example.merge(sub, on='listing_id', how='left')
sub_example['due_date'] = pd.to_datetime(sub_example['due_date'])
sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days
test_prob = sub_example[prob_cols].values
test_labels = sub_example['days'].values
test_prob = [test_prob[i][test_labels[i]] for i in range(test_prob.shape[0])]
sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv('sub.csv', index=False)
