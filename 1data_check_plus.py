# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 1data_check_plus.py
# @time  : 2019/6/13
"""
文件说明：组合为特征后的数据探查
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

# oripath = "F:/数据集/1906拍拍/"
# inpath = "F:/数据集处理/1906拍拍/"
# outpath = "F:/项目相关/1906拍拍/out/"
oripath = "/data/dev/lm/paipai/ori_data/"
inpath = "/data/dev/lm/paipai/feature/"
outpath = "/data/dev/lm/paipai/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
print(df_basic.shape)
df_train = pd.read_csv(open(inpath + "feature_basic_train.csv", encoding='utf8'))
print(df_train.shape)
df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs.csv", encoding='utf8'))
print(df_behavior_logs.shape)
df_listing_info = pd.read_csv(open(inpath + "feature_listing_info.csv", encoding='utf8'))
print(df_listing_info.shape)
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs.csv", encoding='utf8'))
print(df_repay_logs.shape)
df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag.csv", encoding='utf8'))
print(df_user_info_tag.shape)
df_other = pd.read_csv(open(inpath + "feature_other.csv", encoding='utf8'))
print(df_other.shape)
#合并所有特征
df = df_basic.merge(df_train,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_behavior_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_listing_info,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_repay_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_user_info_tag,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_other,how='left',on=['user_id','listing_id','auditing_date'])
print(df.shape)
#调整多分类y
df["y_date_diff"] = df["y_date_diff"].replace(-1,32) #-1~31 调整为0~32
df["y_date_diff_bin"] = df["y_date_diff_bin"].replace(-1,9)
df["y_date_diff_bin3"] = df["y_date_diff_bin3"].replace(-1,2)

train = df[df["auditing_date"]<='2018-12-31']
train['repay_amt'] = train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train["y_date_diff"]=train["y_date_diff"].astype(int)
train["y_date_diff_bin"]=train["y_date_diff_bin"].astype(int)
train["y_date_diff_bin3"]=train["y_date_diff_bin3"].astype(int)
train = train.loc[train["y_date_diff"].isin([0,32])==False]
test = df[df["auditing_date"]>='2019-01-01']
print(train.shape)
print(test.shape)