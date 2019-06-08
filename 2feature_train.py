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
from dateutil.relativedelta import relativedelta

path = "F:/数据集/1906拍拍/"
outpath = "F:/数据集处理/1906拍拍/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_basic.csv", encoding='utf8'))
basic["auditing_date"] = pd.to_datetime(basic["auditing_date"])
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic = basic.sort_values(["user_id", "listing_id"])
# 近3/6个月特征
agg = {
    "due_amt": ["count", "max", "mean"],
    'y_date_diff': ['max', 'min', 'mean'],
    'y_date_diff_bin': ['max', 'min', 'mean'],
    'y_is_last_date': ['sum'],
    'y_is_overdue': ["sum"]
}
#左连接筛选卡时间
basic_info = basic[basic.columns[:-2]]
basic_info = basic_info.rename(columns={'listing_id':'listing_id_info','auditing_date':'auditing_date_info'})
basic = basic[["user_id","listing_id","auditing_date","auditing_date_last3","auditing_date_last6"]]
basic_union = basic.merge(basic_info,how='left',on='user_id')
print(basic_union.shape)
for month in [3,6]:
    print(month)

    basic_tmp = basic_union.loc[(basic_union["auditing_date_info"]<basic_union["auditing_date"])&(
            basic_union["auditing_date_info"]>=basic_union["auditing_date_last{}".format(month)])]
    basic_tmp = basic_tmp.groupby(["user_id","listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['basic_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"basic_last{}_user_id_".format(month):"user_id","basic_last{}_listing_id_".format(month):"listing_id"})
    basic = basic.merge(basic_tmp,how='left',on=["user_id","listing_id"])

del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
basic.to_csv(outpath+'feature_basic_train.csv',index=None)
