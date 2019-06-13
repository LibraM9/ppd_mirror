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

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
inpath = "/data/dev/lm/paipai/feature/"
outpath = "/data/dev/lm/paipai/feature/"

df_train = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs.csv", encoding='utf8'))

df = df_train.merge(df_repay_logs,how='left',on=["user_id","listing_id","auditing_date"])
# 当前金额站过去1/2/3还款金额的比值
df["other_due_ratio_order0"] = df["due_amt"]/df["repay_logs_order0_repay_amt_sum"]
df["other_due_ratio_order1"] = df["due_amt"]/df["repay_logs_order1_repay_amt_sum"]
df["other_due_ratio_order2"] = df["due_amt"]/df["repay_logs_order2_repay_amt_sum"]
df["other_due_ratio_order3"] = df["due_amt"]/df["repay_logs_order3_repay_amt_sum"]
df["other_due_ratio_last1"] = df["due_amt"]/df["repay_logs_last1_repay_amt_sum"]
df["other_due_ratio_last2"] = df["due_amt"]/df["repay_logs_last2_repay_amt_sum"]
df["other_due_ratio_last3"] = df["due_amt"]/df["repay_logs_last3_repay_amt_sum"]

feature = ["user_id","listing_id","auditing_date","other_due_ratio_order0"
           ,"other_due_ratio_order1","other_due_ratio_order2","other_due_ratio_order3"
           ,"other_due_ratio_last1","other_due_ratio_last2","other_due_ratio_last3"]

df[feature].to_csv(outpath+'feature_other.csv',index=None)