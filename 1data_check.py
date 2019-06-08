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
train = pd.read_csv(open(path+"train.csv",encoding='utf8')) #100W
test = pd.read_csv(open(path+"test.csv",encoding='utf8')) #13W
submission = pd.read_csv(open(path+"submission.csv",encoding='utf8')) #398W
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'))
user_info = pd.read_csv(open(path+"user_info.csv",encoding='utf8'))
user_taglist = pd.read_csv(open(path+"user_taglist.csv",encoding='utf8'))
# user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8')) #1G
#121080/130000在test中
user_repay_logs = pd.read_csv(open(path+"user_repay_logs.csv",encoding='utf8')) #900M

#test中出现在train中的个数25537/130000
train_id = train["user_id"].unique()
print(test[test.user_id.isin(train_id)==True].shape)

#还款日分布
# train["month"] = train["auditing_date"].apply(lambda x:int(x[5:7]))
# train = train[train.month.isin([1,3,5,7,8,10,12])]
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
#还款日期分布 0.11719未还款 0.408当天还款 0.12108提前一天 0.05943 提前2 0.05640 提前3
data_cnt.plot(kind='bar')

#还款次数分布 一个订单仅还款1次
repay_cnt = train.groupby(["user_id","listing_id"],as_index=False)["due_amt"].count()
repay_cnt = repay_cnt["due_amt"].value_counts() / train.shape[0]
repay_cnt.plot(kind='bar')

#还款的人是否全额还款 /全部全额还款
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

#一人多标的详情
x = train[train["user_id"]==893965].sort_values("listing_id")

#循环做出用户在时间节点之前的特征
train = train.sort_values("listing_id")

#用户行为 2017-07-05~2019-03-30 似乎没有必然联系
user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8'))
#2017-07-05 00:39:51~2019-03-30 23:59:59
print(user_behavior_logs["behavior_time"].max())
print(user_behavior_logs["behavior_time"].min())
time = user_behavior_logs["behavior_time"].apply(lambda x: x[:10])
time = time.unique()
time.sort()

#listing_info表时间范围2016-07-05~2019-03-31
print(listing_info["auditing_date"].max())
print(listing_info["auditing_date"].min())

#listing_info id所属城市
print(user_info["id_city"].value_counts())

#user_taglist 用户画像5986个
user_taglist["taglist"] = user_taglist.taglist.apply(lambda x:x.split("|"))
ans = set([])
for i in range(user_taglist.shape[0]):
    print(i)
    ans = ans.union(set(user_taglist["taglist"][i]))
print(len(ans))

