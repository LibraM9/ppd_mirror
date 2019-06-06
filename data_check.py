# -*- coding: utf-8 -*-
#@author: limeng
#@file: data_check.py
#@time: 2019/6/5 19:50
"""
文件说明：数据探查
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = "F:/数据集/1906拍拍/"
train = pd.read_csv(open(path+"train.csv",encoding='utf8'))
test = pd.read_csv(open(path+"test.csv",encoding='utf8'))
submission = pd.read_csv(open(path+"submission.csv",encoding='utf8'))
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'))
user_info = pd.read_csv(open(path+"user_info.csv",encoding='utf8'))
user_taglist = pd.read_csv(open(path+"user_taglist.csv",encoding='utf8'))
# user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8')) #1G
# user_repay_logs = pd.read_csv(open(path+"user_repay_logs.csv",encoding='utf8')) #900M

#还款日分布
def date_trans(x):
    try:
        return pd.to_datetime(x)
    except:
        return 'N'
train["due_date_d"] = pd.to_datetime(train["due_date"])
train["repay_date_d"] = train["repay_date"].apply(date_trans)
def date_diff(df):
    try:
        return (df["due_date_d"]-df["repay_date_d"]).days
    except:
        return -1
train["date_diff"] = train.apply(date_diff, axis=1)
data_cnt = train["date_diff"].value_counts() / train.shape[0]
data_cnt = data_cnt.sort_index()
#还款日期分布
data_cnt.plot(kind='bar')

#还款次数分布
repay_cnt = train.groupby(["user_id","listing_id"],as_index=False)["due_amt"].count()
repay_cnt = repay_cnt["due_amt"].value_counts() / train.shape[0]
repay_cnt.plot(kind='bar')

#还款的人是否全额还款
train_amt = train[train["repay_amt"]!='\\N']
train_amt["equal"] = (train_amt["due_amt"].apply(lambda x:round(x,4)).astype(float)==train_amt["repay_amt"].astype(float))
print(train_amt[train_amt.equal==False])

# train 2018-01-01~2018-12-31
date_train = train.auditing_date.unique()
date_train.sort()
print(date_train)
#test 2019-02-01~2019-03-31
date_test = test.auditing_date.unique()
date_test.sort()
print(date_test)

#人均order数量 680852/1 115797/2 22020/3 4000/4
print(train.shape)
print(train.user_id.nunique())
print(train.listing_id.nunique())
date_user = train.groupby(["user_id"],as_index=False)["listing_id"].count()
date_user = date_user.sort_values("listing_id")
user_cnt = date_user["listing_id"].value_counts()
user_cnt.plot("bar")
plt.show()

#一人多标的
x = train[train["user_id"]==893965].sort_values("listing_id")

#循环做出用户在时间节点之前的特征
train = train.sort_values("listing_id")

#用户行为 2017-07-05~2019-03-30
user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8'))
time = user_behavior_logs["behavior_time"].apply(lambda x: x[:10])
time = time.unique()
time.sort()