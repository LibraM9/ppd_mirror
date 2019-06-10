# -*- coding: utf-8 -*-
#@author: limeng
#@file: 2feature_listing_info.py
#@time: 2019/6/8 8:39
"""
文件说明：对feature_listing_info特征构造
"""
import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import relativedelta

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'),parse_dates=['auditing_date'])
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic["auditing_date_last9"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+9))
basic["auditing_date_last12"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+12))
basic = basic.sort_values(["user_id", "listing_id"])
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'),parse_dates=['auditing_date'])

# 近n个月特征
agg = {
    "term": ["count", "max", "min","mean","std"],
    "rate":["max","min","mean","std"],
    "principal": ["max", "min", "mean","std"],
}
#左连接筛选卡时间
listing_info.columns = ["user_id","listing_id_info","auditing_date_info","term","rate","principal"]
basic_union = basic.merge(listing_info,how='left',on='user_id')
print(basic_union.shape)
for month in [3,6,9,12]:
    print(month)

    basic_tmp = basic_union.loc[(basic_union["auditing_date_info"]<basic_union["auditing_date"])&(
            basic_union["auditing_date_info"]>=basic_union["auditing_date_last{}".format(month)])]
    if month == 12:
        basic_tmp["date_diff"]=(basic_tmp["auditing_date"]-basic_tmp["auditing_date_info"]).apply(lambda x:int(x.days))
        agg['date_diff']=["max","min","mean","std"]
    basic_tmp = basic_tmp.groupby(["user_id","listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['listing_info_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"listing_info_last{}_user_id_".format(month):"user_id","listing_info_last{}_listing_id_".format(month):"listing_id"})
    basic = basic.merge(basic_tmp,how='left',on=["user_id","listing_id"])
#当前标的属性
listing_info.columns = ['user_id','listing_id','auditing_date','listing_info_term','listing_info_rate','listing_info_principal']
basic = basic.merge(listing_info,how='left',on=['user_id','listing_id','auditing_date'])

#全部历史最近一次借款距当前最小天/最大天
basic_union = basic_union.loc[basic_union["auditing_date_info"]<basic_union["auditing_date"]]
basic_union["date_diff"] = (basic_union["auditing_date"]-basic_union["auditing_date_info"]).dt.days
date_diff_cnt = basic_union.groupby(["user_id","listing_id"],as_index=False).agg({'date_diff':["count","max",'min','mean','std']})
date_diff_cnt.columns = ['listing_info_date_diff_' + i[0] + '_' + i[1] for i in date_diff_cnt.columns]
date_diff_cnt = date_diff_cnt.rename(columns={"listing_info_date_diff_user_id_":"user_id","listing_info_date_diff_listing_id_":"listing_id"})
basic = basic.merge(date_diff_cnt,how='left',on=["user_id","listing_id"])

del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
del basic["auditing_date_last9"]
del basic["auditing_date_last12"]
basic.to_csv(outpath+'feature_listing_info.csv',index=None)
