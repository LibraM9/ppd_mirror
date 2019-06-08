# -*- coding: utf-8 -*-
#@author: limeng
#@file: rule_lyz.py
#@time: 2019/6/7 21:20
"""
文件说明：
"""
import pandas as pd
import numpy as np
import time
import datetime
import tqdm

#计算两个日期相差天数，自定义函数名，和两个日期的变量名。
def Caltime(date1,date2):
    if date1 =="\\N":
        return -1
    #%Y-%m-%d为日期格式，其中的-可以用其他代替或者不写，但是要统一，同理后面的时分秒也一样；可以只计算日期，不计算时间。
    #date1=time.strptime(date1,"%Y-%m-%d %H:%M:%S")
    #date2=time.strptime(date2,"%Y-%m-%d %H:%M:%S")
    date1=time.strptime(date1,"%Y-%m-%d")
    date2=time.strptime(date2,"%Y-%m-%d")
    #根据上面需要计算日期还是日期时间，来确定需要几个数组段。下标0表示年，小标1表示月，依次类推...
    #date1=datetime.datetime(date1[0],date1[1],date1[2],date1[3],date1[4],date1[5])
    #date2=datetime.datetime(date2[0],date2[1],date2[2],date2[3],date2[4],date2[5])
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    date2=datetime.datetime(date2[0],date2[1],date2[2])
    #返回两个变量相差的值，就是相差天数
    return (date2-date1).days

def get_date(date1, days):
    date1=time.strptime(date1,"%Y-%m-%d")
    date1=datetime.datetime(date1[0],date1[1],date1[2])
    return str(date1-datetime.timedelta(days=days))[0:10]
path = "F:/数据集/1906拍拍/"
train_data = pd.read_csv(open(path+"train.csv",encoding='utf8')) #100W
testt_data = pd.read_csv(open(path+"test.csv",encoding='utf8')) #13W
train_data['repay_amt'] = train_data.repay_amt.map(lambda x: np.nan if x == "\\N" else float(x))
train_data['auditing_month'] = train_data.auditing_date.map(lambda x:x[0:7])
train_data['early_repay_days'] = train_data.apply(lambda x: Caltime(x[5],x[3]), axis=1)

result = testt_data[['listing_id','due_date','auditing_date','due_amt']]
result.columns = ['listing_id','repay_date','auditing_date','repay_amt']

repay_days_distribution = train_data.groupby('early_repay_days').size().reset_index().rename(columns={0:"sums"})

repay_days_distribution['rate'] = repay_days_distribution['sums']/sum(repay_days_distribution['sums'])
new_result = {
    'listing_id': [],
    'repay_date': [],
    'repay_amt': []
}

for i, row in tqdm.tqdm_notebook(enumerate(result.values)):
    tmp_listing_id = row[0]
    tmp_repay_date = row[1]
    tmp_auditing_date = row[2]
    tmp_repay_amt = row[3]

    for j in range(Caltime(tmp_auditing_date, tmp_repay_date)):
        new_result['listing_id'].append(tmp_listing_id)
        new_result['repay_date'].append(get_date(tmp_repay_date, j))
        counts = repay_days_distribution[repay_days_distribution.early_repay_days == j]['sums']
        sums = \
        repay_days_distribution[repay_days_distribution.early_repay_days <= Caltime(tmp_auditing_date, tmp_repay_date)][
            'sums'].sum()
        rates = counts / sums
        new_result['repay_amt'].append(tmp_repay_amt * rates.values[0])