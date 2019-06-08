# -*- coding: utf-8 -*-
#@author: limeng
#@file: 2feature_behavior_logs.py
#@time: 2019/6/8 16:19
"""
文件说明：user_behavior_logs
"""
import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import relativedelta

path = "F:/数据集/1906拍拍/"
outpath = "F:/数据集处理/1906拍拍/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'))
basic["auditing_date"] = pd.to_datetime(basic["auditing_date"])
basic["auditing_date_last7"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+7))
#最后还款日
basic["dead_line"] = basic["auditing_date"].apply(lambda x: x + relativedelta(months=+1))
basic = basic.sort_values(["user_id", "listing_id"])

user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8')) #1G

basic_union = basic.merge(user_behavior_logs,how='left',on='user_id')
basic_union["behavior_time"]=pd.to_datetime(basic_union["behavior_time"])
basic_union = basic_union.loc[(basic_union["behavior_time"]>=basic_union["auditing_date_last7"])&(
basic_union["behavior_time"]<=basic_union["dead_line"]
)]

basic_union["behavior_hour"] = basic_union["behavior_time"].apply(lambda x:int(str(x)[11:13]))
basic_union["behavior_time"] = basic_union["behavior_time"].apply(lambda x:str(x)[:10])
basic_union["behavior_time"] = pd.to_datetime(basic_union["behavior_time"])

#订单前7天行为
basic_union_last = basic_union.loc[(basic_union["behavior_time"] >= basic_union["auditing_date_last7"]) & (
basic_union["behavior_time"]<basic_union["auditing_date"]
)]

def daytime(df):
    return df[(df>=8)&(df<22)].count()
def nighttime(df):
    return df[(df<8)|(df>=22)].count()
def behavior1(df):
    return df[df==1].count()
def behavior2(df):
    return df[df==2].count()
def behavior3(df):
    return df[df==3].count()
agg = {
    "behavior_hour":[daytime,nighttime,'count'],
    "behavior_type":[behavior1,behavior2,behavior3]
}
basic_union_last = basic_union_last.groupby(["user_id", "listing_id"], as_index=False).agg(agg)
basic_union_last.columns = ['behavior_last7_'+ i[0] + '_' + i[1] for i in basic_union_last.columns]
basic_union_last = basic_union_last.rename(columns={'behavior_last7_user_id_':"user_id",'behavior_last7_listing_id_':"listing_id"})

basic = basic[["user_id","listing_id","auditing_date"]].merge(basic_union_last,how='left',on=["user_id","listing_id"])
basic.to_csv(outpath+'feature_behavior_logs.csv',index=None)

#todo 订单后每天行为
basic_union_last = basic_union.loc[(basic_union["behavior_time"] >= basic_union["auditing_date"]) & (
basic_union["behavior_time"]<=basic_union["deadline"]
)]
for i in range(31):
    print(i)
    basic_union["auditing_date_last{}".format(i)] = basic_union["auditing_date"].apply(lambda x: x + relativedelta(days=+i))
