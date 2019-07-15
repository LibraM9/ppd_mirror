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

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/home/dev/lm/paipai/ori_data/"
outpath = "/home/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'),parse_dates=['auditing_date'])
basic["auditing_date_last1"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+1))
basic["auditing_date_last2"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+2))
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic["auditing_date_last12"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+12))

basic = basic.sort_values(["user_id", "listing_id"])
#4216576
user_repay_logs = pd.read_csv(open(path+"user_repay_logs.csv",encoding='utf8'),parse_dates=['due_date','repay_date'])

listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'),parse_dates=['auditing_date'])
# repay_log_df = user_repay_logs.loc[user_repay_logs['order_id'] ==1] #276W数据
user_repay_logs = user_repay_logs.merge(listing_info[['listing_id','auditing_date']],on = 'listing_id',how = 'left')
user_repay_logs["due_duration"] = (user_repay_logs['due_date'] - user_repay_logs['auditing_date']).dt.days#账单时长
print("原始维度",user_repay_logs.shape)
#删除脏数据
ignore_list = user_repay_logs[(user_repay_logs["due_duration"].isin([28,30,31])==False)&(user_repay_logs["order_id"]==1)]["listing_id"].tolist()
user_repay_logs = user_repay_logs[user_repay_logs["listing_id"].isin(ignore_list)==False].reset_index(drop=True)
print("删除后的维度",user_repay_logs.shape)
del user_repay_logs["auditing_date"]
del user_repay_logs["due_duration"]

user_repay_logs["date_diff"] = (user_repay_logs["due_date"]-user_repay_logs["repay_date"]).dt.days #提前天数
user_repay_logs["day_of_week"] = user_repay_logs["repay_date"].dt.dayofweek #周一为0，周日为6
user_repay_logs["day_of_month"] = user_repay_logs["repay_date"].dt.day #还款日期
user_repay_logs.loc[user_repay_logs["repay_date"]==pd.to_datetime('2200-01-01'),'day_of_week']=np.nan #置为空
user_repay_logs.loc[user_repay_logs["repay_date"]==pd.to_datetime('2200-01-01'),'day_of_month']=np.nan
user_repay_logs=user_repay_logs.sort_values(["user_id", "listing_id","repay_date","date_diff"])

#用账单日替换逾期日
user_repay_logs.loc[user_repay_logs["date_diff"]<0,'repay_date']=user_repay_logs.loc[user_repay_logs["date_diff"]<0,'due_date']
#卡时间，我们只能了解借款之前的信息
user_repay_logs = user_repay_logs.rename(columns={"listing_id":"listing_id_repay_logs"})
basic_union = basic.merge(user_repay_logs[["user_id",'listing_id_repay_logs',"order_id","due_date",'repay_amt',"repay_date","date_diff","day_of_week","day_of_month"]],how='left',on=["user_id"])
basic_union = basic_union.loc[basic_union["repay_date"]<basic_union["auditing_date"]]#部分无数据故数据量少于113W
#按最近日期排序,同一个日期，提前还款天数差最大的排前面
basic_union=basic_union.sort_values(["user_id", "listing_id","repay_date","date_diff"],ascending=(True, True,True,False))
basic_union['rank']=basic_union.groupby(["user_id","listing_id"])["repay_date"].rank(ascending=False,method='first')
#给逾期天数定为-20
basic_union['date_diff'] = basic_union['date_diff'].apply(lambda x:x if x>=0 else -20)
#加入期数
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'),parse_dates=['auditing_date'])
listing_info = listing_info.rename(columns={"listing_id":"listing_id_repay_logs"})
basic_union = basic_union.merge(listing_info[["user_id","listing_id_repay_logs","term"]],how='left',on=["user_id","listing_id_repay_logs"])
# 100天内最近一次账单还款距当前借款日期的时间,金额，提前还款天数
basic_union["last_diff"] = (basic_union["auditing_date"]-basic_union["repay_date"]).apply(lambda x:int(x.days))#还款日距当前订单时间
basic_rank = basic_union.loc[(basic_union["rank"]==1)&(basic_union["last_diff"]<=100)][["user_id","listing_id",'last_diff','repay_amt','date_diff',"day_of_week","day_of_month","order_id"]]
basic_rank = basic_rank.rename(columns={'last_diff':"repay_logs_last_diff100",'repay_amt':'repay_logs_repay_amt_last100',
                                        'date_diff':"repay_logs_date_diff100", "day_of_week":"repay_logs_day_of_week100",
                                        "day_of_month":"repay_logs_day_of_month100","order_id":"repay_logs_order_id100"})
basic = basic.merge(basic_rank,how='left',on=["user_id","listing_id"])
# 借款日后一个月内要还款的金额、笔数、及可能的时间
basic_month_pre = basic_union.loc[(basic_union["repay_date"]<basic_union["auditing_date"])&(
            basic_union["repay_date"]>=basic_union["auditing_date_last1"])]
basic_month_pre["last_diff"] = (basic_month_pre["auditing_date"]-basic_month_pre["repay_date"]).apply(lambda x:int(x.days))#还款日期距当前订单时间
basic_month_pre = basic_month_pre.loc[basic_month_pre["order_id"]<basic_month_pre["term"]]
prepayment_list = set(basic_month_pre[basic_month_pre["date_diff"]>31]["listing_id_repay_logs"].tolist())
basic_month_pre2 = basic_month_pre.loc[basic_month_pre["listing_id_repay_logs"].isin(prepayment_list)==False] #删除提前还款的订单
prepayment_list2 = basic_month_pre2.loc[(basic_month_pre2["due_date"]>basic_month_pre2["auditing_date"])|(
        basic_month_pre2["due_date"]<=basic_month_pre2["auditing_date_last1"])]["listing_id_repay_logs"].tolist()
basic_month_pre3 = basic_month_pre2.loc[basic_month_pre2["listing_id_repay_logs"].isin(set(prepayment_list2)) == False]  # 删除提前还款的订单
basic_month_pre3["order_diff"] = (basic_month_pre["auditing_date"]-basic_month_pre["due_date"]).apply(lambda x:int(x.days))#其他账单日距当前订单时间

agg = {
    "repay_amt":["max","min","sum",'std','mean'], #下个月应还款统计
    "last_diff":["count","max","min"], #下个月还款笔数，离账单日时间
    "order_diff":["max","min"]
}
basic_month_pre_union = basic_month_pre3.groupby(["user_id", "listing_id"], as_index=False).agg(agg)
basic_month_pre_union.columns = ['repay_logs_monthpre_'+ col[0] + '_' + col[1] for col in basic_month_pre_union.columns]
basic_month_pre_union = basic_month_pre_union.rename(columns={'repay_logs_monthpre_user_id_':"user_id",'repay_logs_monthpre_listing_id_':"listing_id"})
basic = basic.merge(basic_month_pre_union,how='left',on=["user_id","listing_id"])
#上个月是否有清贷记录
basic_clear_loan= basic_union.loc[(basic_union["repay_date"]<basic_union["auditing_date"])&(
            basic_union["repay_date"]>=basic_union["auditing_date_last1"])]
basic_clear_loan = basic_clear_loan.loc[(basic_clear_loan["order_id"]==basic_clear_loan["term"])&(basic_clear_loan["date_diff"]>=0)]
basic_clear_loan_union = basic_clear_loan.groupby(["user_id", "listing_id"], as_index=False).agg({"term":["count"]})
basic_clear_loan_union.columns = ["user_id","listing_id","repay_logs_last1_clear_loan_cnt"]
basic = basic.merge(basic_clear_loan_union,how='left',on=["user_id","listing_id"])
#提前输出
# del basic["auditing_date_last1"]
# del basic["auditing_date_last2"]
# del basic["auditing_date_last3"]
# del basic["auditing_date_last6"]
# del basic["auditing_date_last12"]
# basic.to_csv(outpath+'feature_repay_logs_plus0702.csv',index=None)
#聚合
#逾期次数
def overdue_cnt(df):
    return df[df<0].count()
#逾期占比
def overdue_ratio(df):
    return df[df<0].count()/df.count()
#0占比
def diff0_ratio(df):
    return df[df==0].count()/df.count()
#1占比
def diff1_ratio(df):
    return df[df==1].count()/df.count()
#2占比
def diff2_ratio(df):
    return df[df==2].count()/df.count()
#3占比
def diff3_ratio(df):
    return df[df==3].count()/df.count()
#周一还款
def week0_cnt(df):
    return df[df==0].count()
#周五还款
def week4_cnt(df):
    return df[df==4].count()
#周六还款
def week5_cnt(df):
    return df[df==5].count()
#周日还款
def week6_cnt(df):
    return df[df==6].count()
#1号还款
def day1_cnt(df):
    return df[df==1].count()
#5号还款
def day5_cnt(df):
    return df[df==5].count()
#6号还款
def day6_cnt(df):
    return df[df==6].count()
#10号还款
def day10_cnt(df):
    return df[df==10].count()
#15号还款
def day15_cnt(df):
    return df[df==15].count()
#16号还款
def day16_cnt(df):
    return df[df==16].count()
#20号还款
def day20_cnt(df):
    return df[df==20].count()
#21号还款
def day21_cnt(df):
    return df[df==21].count()
#25号还款
def day25_cnt(df):
    return df[df==25].count()
#26号还款
def day26_cnt(df):
    return df[df==26].count()
agg = {
    'repay_amt':['max','min','sum','std','mean'],
    'date_diff':['count','mean','min','max','std',overdue_cnt,overdue_ratio,diff0_ratio,diff1_ratio,diff2_ratio,diff3_ratio],
    'day_of_week':[week0_cnt,week4_cnt,week5_cnt,week6_cnt],
    'day_of_month':[day1_cnt,day5_cnt,day6_cnt,day10_cnt,day15_cnt,day16_cnt,day20_cnt,
                    day21_cnt,day25_cnt,day26_cnt]
} #提前还款情况、周几还款情况、几号还款情况
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

#近1/2/3/6/12个月 还款情况
for month in [1,2,3,6,12]:
    print(month)

    basic_tmp = basic_union.loc[(basic_union["repay_date"]<basic_union["auditing_date"])&(
            basic_union["repay_date"]>=basic_union["auditing_date_last{}".format(month)])]
    basic_tmp = basic_tmp.groupby(["user_id","listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['repay_logs_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"repay_logs_last{}_user_id_".format(month):"user_id","repay_logs_last{}_listing_id_".format(month):"listing_id"})
    basic = basic.merge(basic_tmp,how='left',on=["user_id","listing_id"])

# 历史还款记录距今时间最大/最小值/均值
basic_union["auditing_date_diff"] = (basic_union["auditing_date"]-basic_union["repay_date"]).dt.days
date_diff_cnt = basic_union.groupby(["user_id","listing_id"],as_index=False).agg({'auditing_date_diff':["max",'min','mean','std']})
date_diff_cnt.columns = ['repay_logs_auditing_date_diff_' + i[0] + '_' + i[1] for i in date_diff_cnt.columns]
date_diff_cnt = date_diff_cnt.rename(columns={"repay_logs_auditing_date_diff_user_id_":"user_id","repay_logs_auditing_date_diff_listing_id_":"listing_id"})
basic = basic.merge(date_diff_cnt,how='left',on=["user_id","listing_id"])

# 1/2/3期账单与历史的比例
basic["repay_logs_order1_repay_amt_sum_ratio"] = basic["repay_logs_order1_repay_amt_sum"]/basic["repay_logs_order0_repay_amt_sum"]
basic["repay_logs_order2_repay_amt_sum_ratio"] = basic["repay_logs_order2_repay_amt_sum"]/basic["repay_logs_order0_repay_amt_sum"]
basic["repay_logs_order3_repay_amt_sum_ratio"] = basic["repay_logs_order3_repay_amt_sum"]/basic["repay_logs_order0_repay_amt_sum"]
basic["repay_logs_order1_date_diff_mean_ratio"] = basic["repay_logs_order1_date_diff_mean"]/basic["repay_logs_order0_date_diff_mean"]
basic["repay_logs_order2_date_diff_mean_ratio"] = basic["repay_logs_order2_date_diff_mean"]/basic["repay_logs_order0_date_diff_mean"]
basic["repay_logs_order3_date_diff_mean_ratio"] = basic["repay_logs_order3_date_diff_mean"]/basic["repay_logs_order0_date_diff_mean"]
# 近1/2/3/6月账单与历史的比例
for i in [1,2,3,6]:
    basic["repay_logs_last{}_repay_amt_sum_his_ratio".format(i)] = basic["repay_logs_last{}_repay_amt_sum".format(i)]/basic["repay_logs_order0_repay_amt_sum"]
    basic["repay_logs_last{}_date_diff_mean_his_ratio".format(i)] = basic["repay_logs_last{}_date_diff_mean".format(i)] / basic["repay_logs_order0_date_diff_mean"]
# 近1月账单与2/3/6/12的比例
for i in [2,3,6,12]:
    basic["repay_logs_last1_repay_amt_sum_last{}_ratio".format(i)] = basic["repay_logs_last1_repay_amt_sum"]/basic["repay_logs_last{}_repay_amt_sum".format(i)]
    basic["repay_logs_last1_date_diff_mean_last{}_ratio".format(i)] = basic["repay_logs_last1_date_diff_mean"]/basic["repay_logs_last{}_date_diff_mean".format(i)]
# 近2月账单与3/6/12的比例
for i in [3,6,12]:
    basic["repay_logs_last2_repay_amt_sum_last{}_ratio".format(i)] = basic["repay_logs_last2_repay_amt_sum"]/basic["repay_logs_last{}_repay_amt_sum".format(i)]
    basic["repay_logs_last2_date_diff_mean_last{}_ratio".format(i)] = basic["repay_logs_last2_date_diff_mean"]/basic["repay_logs_last{}_date_diff_mean".format(i)]
# 近3月账单与6/12的比例
for i in [6,12]:
    basic["repay_logs_last3_repay_amt_sum_last{}_ratio".format(i)] = basic["repay_logs_last3_repay_amt_sum"]/basic["repay_logs_last{}_repay_amt_sum".format(i)]
    basic["repay_logs_last3_date_diff_mean_last{}_ratio".format(i)] = basic["repay_logs_last3_date_diff_mean"]/basic["repay_logs_last{}_date_diff_mean".format(i)]

del basic["auditing_date_last1"]
del basic["auditing_date_last2"]
del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
del basic["auditing_date_last12"]

basic.to_csv(outpath+'feature_repay_logs_plus0702.csv',index=None)


##探查
listing_info = pd.read_csv(open(path+"listing_info.csv",encoding='utf8'),parse_dates=['auditing_date'])
repay_log_df = user_repay_logs.loc[user_repay_logs['order_id'] ==1] #276W数据
repay_log_df = repay_log_df.merge(listing_info[['listing_id','auditing_date']],on = 'listing_id',how = 'left')
((repay_log_df['due_date'] - repay_log_df['auditing_date']).dt.days).value_counts()

