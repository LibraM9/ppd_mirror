# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 2feature_other.py
# @time  : 2019/6/11
"""
文件说明：多表聚合特征
"""
import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import relativedelta

date = "0619"
# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
inpath = "/home/dev/lm/paipai/feature/"
outpath = "/home/dev/lm/paipai/feature/"

df_train = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs{}.csv".format(date), encoding='utf8'))

df = df_train.merge(df_repay_logs,how='left',on=["user_id","listing_id","auditing_date"])
# 当前金额占过去1/2/3还款金额的比值
df["other_due_ratio_order0"] = df["due_amt"]/df["repay_logs_order0_repay_amt_sum"]
df["other_due_ratio_order1"] = df["due_amt"]/df["repay_logs_order1_repay_amt_sum"]
df["other_due_ratio_order2"] = df["due_amt"]/df["repay_logs_order2_repay_amt_sum"]
df["other_due_ratio_order3"] = df["due_amt"]/df["repay_logs_order3_repay_amt_sum"]
df["other_due_ratio_last1"] = df["due_amt"]/df["repay_logs_last1_repay_amt_sum"]
df["other_due_ratio_last2"] = df["due_amt"]/df["repay_logs_last2_repay_amt_sum"]
df["other_due_ratio_last3"] = df["due_amt"]/df["repay_logs_last3_repay_amt_sum"]
df["other_due_ratio_last6"] = df["due_amt"]/df["repay_logs_last6_repay_amt_sum"]
df["other_due_ratio_last12"] = df["due_amt"]/df["repay_logs_last12_repay_amt_sum"]

#user_tag lgb分类结果
user_tag_pred = pd.read_csv(open(inpath + "user_tag_pred.csv", encoding='utf8'))
user_tag_pred.columns = ["listing_id","other_tag_pred_is_last_date","other_tag_pred_is_overdue"]
df = df.merge(user_tag_pred,how='left',on="listing_id")
#user_tag lgb高IV列
user_tag_iv = pd.read_csv(open(inpath + "user_tag_iv.csv", encoding='utf8'))
user_tag_iv.columns = ["other_"+i for i in user_tag_iv.columns[:-1]]+["listing_id"]
df = df.merge(user_tag_iv,how='left',on="listing_id")

feature = ["user_id","listing_id","auditing_date"]
for i in df.columns:
    if i[:5]=="other":
        feature.append(i)

df[feature].to_csv(outpath+'feature_other0619.csv',index=None)