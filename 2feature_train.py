# -*- coding: utf-8 -*-
# @author: limeng
# @file: 2feature_train.py
# @time: 2019/6/8 2:29
"""
文件说明：对train/test进行特征构造
"""
import pandas as pd
import numpy as np
import gc
from scipy.stats import kurtosis #峰度
from dateutil.relativedelta import relativedelta

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_basic.csv", encoding='utf8'), parse_dates=['auditing_date', 'due_date', 'repay_date'])
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic["auditing_date_last9"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+9))
basic = basic.sort_values(["user_id", "listing_id"])

#平均每天还款金额
basic["m_days"] = (basic["due_date"]-basic["auditing_date"]).dt.days
basic["due_amt_every_day"] = basic["due_amt"]/basic["m_days"]
# 近3/6/9个月特征
agg = {
    "due_amt": ["count", "max", "min","mean","std", 'median',"sum"],
    'y_date_diff': ['max', 'min', 'mean',"std",'median'],
    'y_date_diff_bin': ['max', 'min', 'mean',"std",'median'],
    'y_is_last_date': ['sum', 'mean',"std"],
    'y_is_overdue': ["sum", 'mean',"std"]
}
#左连接筛选卡时间
basic_info = basic[['user_id', 'listing_id', 'auditing_date', 'due_date', 'due_amt',
       'repay_date', 'repay_amt', 'auditing_month', 'y_date_diff',
       'y_date_diff_bin', 'y_date_diff_bin3', 'y_is_last_date', 'y_is_overdue']]
basic_info = basic_info.rename(columns={'listing_id':'listing_id_info','auditing_date':'auditing_date_info'})
basic = basic[["user_id","listing_id","auditing_date","auditing_date_last3","auditing_date_last6","auditing_date_last9","m_days","due_amt_every_day"]]
basic_union = basic.merge(basic_info,how='left',on='user_id')
print(basic_union.shape)
for month in [3,6,9]:
    print(month)

    basic_tmp = basic_union.loc[(basic_union["auditing_date_info"]<basic_union["auditing_date"])&(
            basic_union["auditing_date_info"]>=basic_union["auditing_date_last{}".format(month)])]
    basic_tmp = basic_tmp.groupby(["user_id","listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['basic_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"basic_last{}_user_id_".format(month):"user_id","basic_last{}_listing_id_".format(month):"listing_id"})
    basic = basic.merge(basic_tmp,how='left',on=["user_id","listing_id"])

#当前金额占3/6/9金额比值
basic = basic.merge(basic_info[['listing_id_info','due_amt']],how='left',left_on='listing_id',right_on='listing_id_info')
basic["basic_amt_ratio_last3"]=basic["due_amt"]/basic['basic_last3_due_amt_sum']
basic["basic_amt_ratio_last6"]=basic["due_amt"]/basic['basic_last6_due_amt_sum']
basic["basic_amt_ratio_last9"]=basic["due_amt"]/basic['basic_last9_due_amt_sum']

del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
del basic["auditing_date_last9"]
del basic["listing_id_info"]
del basic["due_amt"]
basic.to_csv(outpath+'feature_basic_train.csv',index=None)
