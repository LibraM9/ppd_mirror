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
from scipy.stats import kurtosis  # 峰度
from dateutil.relativedelta import relativedelta

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_basic.csv", encoding='utf8'),
                    parse_dates=['auditing_date', 'due_date', 'repay_date'])
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+3))
basic["auditing_date_last6"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+6))
basic["auditing_date_last9"] = basic["auditing_date"].apply(lambda x: x - relativedelta(months=+9))
basic["basic_day_of_week"] = basic["auditing_date"].dt.dayofweek  # 周一为0，周日为6
basic["basic_day_of_month"] = basic["auditing_date"].dt.day  # 借款日
basic["basic_day_of_week_due"] = basic["due_date"].dt.dayofweek  # 还款周几
basic["basic_day_of_month_due"] = basic["due_date"].dt.day  # 还款日
basic = basic.sort_values(["user_id", "listing_id"])
basic["basic_spring_festival_diff"]=np.nan # 距春节的天数差
basic.loc[(basic["auditing_date"]>=pd.to_datetime("2018-01-01"))&(basic["auditing_date"]<=pd.to_datetime("2018-04-15")),"basic_spring_festival_diff"]=\
    (pd.to_datetime("2018-02-16")-basic.loc[(basic["auditing_date"]>=pd.to_datetime("2018-01-01"))&(basic["auditing_date"]<=pd.to_datetime("2018-04-15"))]["auditing_date"]).dt.days
basic.loc[(basic["auditing_date"]>=pd.to_datetime("2019-01-01"))&(basic["auditing_date"]<=pd.to_datetime("2019-03-31")),"basic_spring_festival_diff"]=\
    (pd.to_datetime("2019-02-05")-basic.loc[(basic["auditing_date"]>=pd.to_datetime("2019-01-01"))&(basic["auditing_date"]<=pd.to_datetime("2019-03-31"))]["auditing_date"]).dt.days
# 平均每天还款金额
basic["basic_m_days"] = (basic["due_date"] - basic["auditing_date"]).dt.days
basic["basic_due_amt_every_day"] = basic["due_amt"] / basic["basic_m_days"]
# 近3/6/9个月特征
agg = {
    "due_amt": ["count", "max", "min", "mean", "std", 'median', "sum"],
    'y_date_diff': ['max', 'min', 'mean', "std", 'median'],
    'y_date_diff_bin': ['max', 'min', 'mean', "std", 'median'],
    'y_is_last_date': ['sum', 'mean', "std"],
    'y_is_overdue': ["sum", 'mean', "std"]
}
# 左连接筛选卡时间
basic_info = basic[['user_id', 'listing_id', 'auditing_date', 'due_amt',
                    'repay_date', 'repay_amt', 'auditing_month', 'y_date_diff',
                    'y_date_diff_bin', 'y_date_diff_bin3', 'y_is_last_date', 'y_is_overdue']]
basic_info = basic_info.rename(columns={'listing_id': 'listing_id_info', 'auditing_date': 'auditing_date_info'})
basic = basic[
    ["user_id", "listing_id", "auditing_date", "auditing_date_last3", "auditing_date_last6", "auditing_date_last9",
     "due_date"
        , "basic_m_days", "basic_due_amt_every_day", "basic_day_of_week", "basic_day_of_month",
     "basic_day_of_week_due", "basic_day_of_month_due","basic_spring_festival_diff"]]
basic_union = basic.merge(basic_info, how='left', on='user_id')
print(basic_union.shape)
for month in [3, 6, 9]:
    print(month)

    basic_tmp = basic_union.loc[(basic_union["auditing_date_info"] < basic_union["auditing_date"]) & (
            basic_union["auditing_date_info"] >= basic_union["auditing_date_last{}".format(month)])]
    basic_tmp = basic_tmp.groupby(["user_id", "listing_id"], as_index=False).agg(agg)
    basic_tmp.columns = ['basic_last{}_'.format(month) + i[0] + '_' + i[1] for i in basic_tmp.columns]
    basic_tmp = basic_tmp.rename(columns={"basic_last{}_user_id_".format(month): "user_id",
                                          "basic_last{}_listing_id_".format(month): "listing_id"})
    basic = basic.merge(basic_tmp, how='left', on=["user_id", "listing_id"])

# 当前金额占3/6/9金额比值
basic = basic.merge(basic_info[['listing_id_info', 'due_amt']], how='left', left_on='listing_id',
                    right_on='listing_id_info')
basic["basic_amt_ratio_last3"] = basic["due_amt"] / basic['basic_last3_due_amt_sum']
basic["basic_amt_ratio_last6"] = basic["due_amt"] / basic['basic_last6_due_amt_sum']
basic["basic_amt_ratio_last9"] = basic["due_amt"] / basic['basic_last9_due_amt_sum']

# 近3月订单数、订单金额、提前还款日期、首逾记录、截止日还款记录占6/9的比例
for i in [6,9]:
    basic["basic_last3_due_amt_count_ratio_last{}".format(i)] = basic["basic_last3_due_amt_count"]/basic["basic_last{}_due_amt_count".format(i)]
    basic["basic_last3_due_amt_sum_ratio_last{}".format(i)] = basic["basic_last3_due_amt_sum"]/basic["basic_last{}_due_amt_sum".format(i)]
    basic["basic_last3_y_date_diff_mean_ratio_last{}".format(i)] = basic["basic_last3_y_date_diff_mean"] / basic["basic_last{}_y_date_diff_mean".format(i)]
    basic["basic_last3_y_is_last_date_sum_ratio_last{}".format(i)] = basic["basic_last3_y_is_last_date_sum"] / basic["basic_last{}_y_is_last_date_sum".format(i)]
    basic["basic_last3_y_is_overdue_sum_ratio_last{}".format(i)] = basic["basic_last3_y_is_overdue_sum"] / basic["basic_last{}_y_is_overdue_sum".format(i)]
# 账单日据1/5/6/10/15/16/20/21/25/26天数差
# 账单日前1/5/6/10/15/16/20/21/25/26星期几
def find_diff(date, day):
    """
    计算date到当月某日的时间差，若为正则返回，若为负则计算和上个月的时间差
    :param date: 当前日期
    :param day: 最近的某一日
    :return: 当前日期到最近的某一日的时间差
    """
    date_last = date - relativedelta(months=+1)
    date1 = pd.to_datetime(str(date)[:8] + day)
    date2 = pd.to_datetime(str(date_last)[:8] + day)
    diff = (date - date1).days
    if diff >= 0:
        return date1
    else:
        return date2


for day in ['01', '05', '06', '10', '15', '16', '20', '21', '25', '26']:
    print(day)
    basic['basic_due_date_to{}'.format(day)] = basic["due_date"].apply(lambda x: find_diff(x, day))

for day in ['01', '05', '06', '10', '15', '16', '20', '21', '25', '26']:
    print(day)  # 周几 相差几天
    basic['basic_due_date_to{}_week'.format(day)] = basic['basic_due_date_to{}'.format(day)].dt.dayofweek
    basic['basic_due_date_to{}_diff'.format(day)] = (
                basic["due_date"] - basic['basic_due_date_to{}'.format(day)]).dt.days
    del basic['basic_due_date_to{}'.format(day)]

del basic["auditing_date_last3"]
del basic["auditing_date_last6"]
del basic["auditing_date_last9"]
del basic["due_date"]
del basic["listing_id_info"]
del basic["due_amt"]
basic.to_csv(outpath + 'feature_basic_train0619.csv', index=None)
