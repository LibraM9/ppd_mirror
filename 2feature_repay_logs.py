# -*- coding: utf-8 -*-
#@author: limeng
#@file: 2feature_repay_logs.py
#@time: 2019/6/8 19:33
"""
文件说明：user_repay_logs
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

basic = basic.sort_values(["user_id", "listing_id"])
user_repay_logs = pd.read_csv(open(path+"user_repay_logs.csv",encoding='utf8'))
user_repay_logs["due_date"] = pd.to_datetime(user_repay_logs["due_date"])
user_repay_logs["repay_date"] = pd.to_datetime(user_repay_logs["repay_date"])
user_repay_logs["date_diff"] = (user_repay_logs["due_date"]-user_repay_logs["repay_date"]).apply(lambda x:int(x.days))
user_repay_logs=user_repay_logs.sort_values(["user_id", "listing_id","repay_date","date_diff"])
#用账单日替换逾期日
user_repay_logs.loc[user_repay_logs["date_diff"]<0,'repay_date']=user_repay_logs.loc[user_repay_logs["date_diff"]<0,'due_date']
#卡时间，我们只能了解借款之前的信息
basic_union = basic.merge(user_repay_logs[["user_id","order_id","repay_date","date_diff"]],how='left',on=["user_id"])
basic_union = basic_union.loc[basic_union["repay_date"]<basic_union["auditing_date"]]
#按最近日期排序
basic_union=basic_union.sort_values(["user_id", "listing_id","repay_date","date_diff"])
basic_union['rank']=basic_union.groupby(["user_id","listing_id"])["repay_date"].rank(ascending=False,method='first')
#给逾期天数定为-10
basic_union['date_diff'] = basic_union['date_diff'].apply(lambda x:x if x>=0 else -10)
#聚合
#逾期次数
def overdue_cnt(df):
    return df[df<0].count()
agg = {
    'date_diff':['count','mean',overdue_cnt]
}
#全部历史数据/历史1期/2期/3期账单还款时间
for i in [0,1,2,3]:
    print(i)
    if i == 0:
    #全部数据
        basic_union_order = basic_union.groupby(["user_id", "listing_id"], as_index=False).agg(agg)
    else:
        basic_union_order = basic_union.loc[basic_union["order_id"]==i].groupby(["user_id", "listing_id"], as_index=False).agg(agg)
    basic_union_order.columns = ['repay_logs_order{}_'.format(i)+ col[0] + '_' + col[1] for col in basic_union_order.columns]
    basic_union_order = basic_union_order.rename(columns={'repay_logs_order{}_user_id_'.format(i):"user_id",'repay_logs_order{}_listing_id_'.format(i):"listing_id"})
    basic = basic.merge(basic_union_order,how='left',on=["user_id","listing_id"])

# 最近一次账单还款时间
basic_rank = basic_union.loc[basic_union["rank"]==1][["user_id","listing_id",'date_diff']]
basic_rank = basic_rank.rename(columns={'date_diff':"repay_logs_date_diff_last"})
basic = basic.merge(basic_rank,how='left',on=["user_id","listing_id"])

del basic["rank"]
basic.to_csv(outpath+'feature_repay_logs.csv',index=None)