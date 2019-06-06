# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : feature_create2.py
# @time  : 2019/6/6
"""
文件说明：
"""
import pandas as pd
import matplotlib.pyplot as plt

path = 'F:/数据集/1906拍拍/'
train = pd.read_csv(open(path+"train.csv",encoding='utf8'))
test = pd.read_csv(open(path+"test.csv",encoding='utf8'))

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