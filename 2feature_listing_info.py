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

path = "F:/数据集/1906拍拍/"
outpath = "F:/数据集处理/1906拍拍/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'))
basic["auditing_date"] = pd.to_datetime(basic["auditing_date"])
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic["auditing_date_last9"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+9))
basic["auditing_date_last12"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+12))
basic = basic.sort_values(["user_id", "listing_id"])
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'))
listing_info["auditing_date"] = pd.to_datetime(listing_info["auditing_date"])

# 近n个月特征
agg = {
    "term": ["count", "max", "min","mean"],
    "rate":["max","min","mean"],
    "principal": ["max", "min", "mean"],
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
        agg['date_diff']=["max","min"]
    basic_tmp = basic_tmp.groupby(["user_id","listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['listing_info_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"listing_info_last{}_user_id_".format(month):"user_id","listing_info_last{}_listing_id_".format(month):"listing_id"})
    basic = basic.merge(basic_tmp,how='left',on=["user_id","listing_id"])

listing_info.columns = ['user_id','listing_id','auditing_date','listing_info_term','listing_info_rate','listing_info_principal']
basic = basic.merge(listing_info,how='left',on=['user_id','listing_id','auditing_date'])
del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
del basic["auditing_date_last9"]
del basic["auditing_date_last12"]
basic.to_csv(outpath+'feature_listing_info.csv',index=None)
# # 逐条筛选卡时间,耗时过长
# def last(basic,listing_info,month):
#     print(month)
#     new_f = []
#     #初始化新字段
#     for key in agg.keys():
#         for func in agg[key]:
#             basic["listing_info_last{}_".format(month) + key + "_" + func] = 0
#             new_f.append("listing_info_last{}_".format(month) + key + "_" + func)
#
#     #前n个月计算
#     global n
#     n = 0
#     def last_cal(df):
#         global n
#         n = n+1
#         print(n)
#         temp = listing_info.loc[(listing_info["user_id"] == df["user_id"]) & (listing_info["auditing_date"] < df["auditing_date"]) & (
#                 listing_info["auditing_date"] >= df["auditing_date_last{}".format(month)])]
#         temp_feature = temp.groupby("user_id", as_index=False).agg(agg)
#         temp_feature.columns = ['listing_info_last{}_'.format(month) + i[0] + '_' + i[1] for i in temp_feature]
#         for col in temp_feature.columns[1:]:
#             try:
#                 df[col]=temp_feature[col].values[0]
#             except:
#                 if col == 'listing_info_last{}_term_count'.format(month):
#                     df[col]=0
#                 else:
#                     df[col]=np.nan
#
#         return df[temp_feature.columns[1:]]
#
#     basic[new_f] = basic.apply(last_cal,axis=1)
#     return basic
#
# for month in [3,6,9,12]:
#     basic = last(basic,listing_info,month)
#     gc.collect()
#
# #当前标的属性
# listing_info.columns = ['user_id','listing_id','auditing_date','listing_info_term','listing_info_rate','listing_info_principal']
# basic = basic.merge(listing_info,how='left',on=['user_id','listing_id','auditing_date'])
# gc.collect()
# basic.to_csv(outpath+'feature_listing_info.csv',index=None)