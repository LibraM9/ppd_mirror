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
# path = "/home/dev/lm/paipai/ori_data/"
train = pd.read_csv(open(path+"train.csv",encoding='utf8')) #100W
test = pd.read_csv(open(path+"test.csv",encoding='utf8')) #13W
submission = pd.read_csv(open(path+"submission.csv",encoding='utf8')) #398W
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'))
user_info = pd.read_csv(open(path+"user_info.csv",encoding='utf8'))
user_taglist = pd.read_csv(open(path+"user_taglist.csv",encoding='utf8'))
user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8')) #1G
#121080/130000在test中
#2016-08-12~2019-03-30
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
plt.show()

#还款的人是否全额还款 /全部全额还款
train_amt = train[train["repay_amt"]!='\\N']
train_amt["equal"] = (train_amt["due_amt"].apply(lambda x:round(x,4)).astype(float)==train_amt["repay_amt"].astype(float))
train_amt["equal"].value_counts().plot.bar()
plt.show()

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

###################构建一个训练集全还款记录，用以求mse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

inpath = "/data/dev/lm/paipai/feature/"
oripath = "/data/dev/lm/paipai/ori_data/"
train = pd.read_csv(open(oripath+"train.csv",encoding='utf8'))
train.loc[train["repay_date"]=='\\N','repay_date']=train.loc[train["repay_date"]=='\\N','due_date']
train["repay_amt"] = train["repay_amt"].replace('\\N',0)
train["auditing_date"] = pd.to_datetime(train["auditing_date"])
train["due_date"] = pd.to_datetime(train["due_date"])
train["repay_date"] = pd.to_datetime(train["repay_date"])
#train mse文件构建,太耗内存无法运行
from dateutil.relativedelta import relativedelta
date_lst = [pd.to_datetime('2018-01-01')+ relativedelta(days=+i) for i in range(365)]
date_lst = date_lst*train.shape[0]
listing_lst = []
n = 0
for id in train["listing_id"]:
    n = n+1
    print(n)
    temp_lst = [id]*365
    listing_lst.extend(temp_lst)
train_date = pd.DataFrame({"listing_id":listing_lst,"repay_date":date_lst})

train_date = train_date.merge(train[["listing_id","auditing_date","due_date"]],how='left',on="listing_id")
train_date = train_date.loc[(train_date['repay_date']>=train_date['auditing_date'])&(train_date['repay_date']<=train_date['due_date'])]

train_date = train_date.merge(train[["listing_id","repay_date","repay_amt"]],how='left',on=["listing_id","repay_date"])
train_date = train_date.fillna(0)
train_date[["listing_id","repay_date","repay_amt"]].to_csv(inpath+'submission_train.csv',index=None)

dic = {0:0.408187,
1:0.121085,
2:0.05943,
3:0.056404,
4:0.026425,
5:0.02138,
6:0.017568,
7:0.014797,
8:0.012993,
9:0.011393,
10:0.009984,
11:0.009002,
12:0.008219,
13:0.007688,
14:0.00692,
15:0.006443,
16:0.006231,
17:0.005832,
18:0.005492,
19:0.005108,
20:0.004788,
21:0.004504,
22:0.004295,
23:0.004197,
24:0.003922,
25:0.003934,
26:0.00393,
27:0.004102,
28:0.004677,
29:0.005645,
30:0.009865,
31:0.008368}

#train/test中大额还款
train["repay_date"] = train["repay_date"].replace("\\N","2020-01-01")
train["due_date"] = pd.to_datetime(train["due_date"])
train["repay_date"] = pd.to_datetime(train["repay_date"])
train = train.sort_values("due_amt",ascending=False)
test = test.sort_values("due_amt",ascending=False)
train["diff"]=(train["due_date"]-train["repay_date"]).dt.days.apply(lambda x:-1 if x<0 else x)
train5000 = train[train["due_amt"]>=5000]
train5000 = train5000.merge(listing_info,how='left',on=["user_id","listing_id","auditing_date"])
x = train5000["diff"].value_counts()/train5000["diff"].shape[0]
test5000 = test[test["due_amt"]>=5000]
test5000 = test5000.merge(listing_info,how='left',on=["user_id","listing_id","auditing_date"])
ans = user_repay_logs[user_repay_logs.user_id.isin(test5000.user_id.values)]
ans = ans.sort_values(["user_id","listing_id","repay_date"])