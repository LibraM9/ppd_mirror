# -*- coding: utf-8 -*-
#@author: limeng
#@file: feature_create.py
#@time: 2019/6/5 22:26
"""
文件说明：构建各种Y指标
"""
import pandas as pd
import numpy as np

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
train = pd.read_csv(open(path+"train.csv",encoding='utf8')) #100W
test = pd.read_csv(open(path+"test.csv",encoding='utf8')) #13W
test["repay_date"] = ""
test["repay_amt"] = ""
basic = pd.concat([train,test],axis=0)

basic["auditing_month"] = basic["auditing_date"].apply(lambda x:int(x[5:7]))
basic["repay_date"] = basic["repay_date"].replace("\\N","2020-01-01")
basic["due_date_d"] = pd.to_datetime(basic["due_date"])
basic["repay_date_d"] = pd.to_datetime(basic["repay_date"])
def date_diff(num):
    try:
        if num <0:
            return -1
        elif num>=0:
            return int(num)
    except:
        return np.nan
#多分类问题 未还款-1 还款0~31
basic["y_date_diff"] = (basic["due_date_d"]-basic["repay_date_d"]).apply(lambda x:x.days)
basic["y_date_diff"] = basic["y_date_diff"].apply(date_diff)
#多分类问题 未还款-1 /当天还款 0/ 1 2 3 (1)/4 5 6 7(2)/8 9 10 11 (3)/12 13 14 15(4)/
#16 17 18 19(5)/20 21 22 23(6)/24 25 26 27 (7)/28 29 30 31 (8)
def bin(num):
    if num==-1:
        return -1
    elif num == 0:
        return 0
    elif num in [1, 2, 3]:
        return 1
    elif num in [4, 5, 6, 7]:
        return 2
    elif num in [8, 9, 10, 11]:
        return 3
    elif num in [12, 13, 14, 15]:
        return 4
    elif num in [16, 17, 18, 19]:
        return 5
    elif num in [20, 21, 22, 23]:
        return 6
    elif num in [24, 25, 26, 27]:
        return 7
    elif num in [28, 29, 30, 31]:
        return 8
    else:
        return np.nan
basic["y_date_diff_bin"] = basic["y_date_diff"].apply(bin)

#多分类问题 未还款/当天还款/提前还款
def bin3(num):
    if num==-1:
        return -1
    elif num == 0:
        return 0
    elif num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
                 , 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        return 1
    else:
        return np.nan
basic["y_date_diff_bin3"] = basic["y_date_diff"].apply(bin3)

#二分类问题 是否在账单日还款
basic["y_is_last_date"] = basic["y_date_diff"].apply(lambda x:1 if x==0 else 0)
#二分类问题 是否逾期
basic["y_is_overdue"] = basic["y_date_diff"].apply(lambda x:1 if x==-1 else 0)

del basic["due_date_d"]
del basic["repay_date_d"]
basic.to_csv(outpath+'feature_basic.csv',index=None)