# -*- coding: utf-8 -*-
#@author: limeng
#@file: 2feature_behavior_logs.py
#@time: 2019/6/8 16:19
"""
文件说明：user_behavior_logs
"""
import pandas as pd
import numpy as np
import gc
from dateutil.relativedelta import relativedelta

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'),parse_dates=['auditing_date'])
basic["auditing_date_last1"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+1))
basic["auditing_date_last3"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+3))
basic["auditing_date_last7"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+7))
basic["auditing_date_last15"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+15))
basic["auditing_date_last30"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+30))
basic["auditing_date_last60"] = basic["auditing_date"].apply(lambda x: x - relativedelta(days=+60))

basic = basic.sort_values(["user_id", "listing_id"])
#最后还款日######
basic["dead_line"] = basic["auditing_date"].apply(lambda x: x + relativedelta(months=+1))
basic = basic.sort_values(["user_id", "listing_id"])
################
user_behavior_logs = pd.read_csv(open(path+"user_behavior_logs.csv",encoding='utf8'),parse_dates=['behavior_time']) #1G
# user_behavior_logs["behavior_hour"] = user_behavior_logs["behavior_time"].apply(lambda x: int(x[11:13]))
# user_behavior_logs["behavior_time"] = user_behavior_logs["behavior_time"].apply(lambda x: x[:10])
# user_behavior_logs["behavior_time"] = pd.to_datetime(user_behavior_logs["behavior_time"])

basic_union = basic.merge(user_behavior_logs,how='left',on='user_id')
basic_union = basic_union.loc[(basic_union["behavior_time"]>=basic_union["auditing_date_last60"])&(
                basic_union["behavior_time"]<=basic_union["dead_line"])]

basic_union["behavior_hour"] = basic_union["behavior_time"].apply(lambda x:int(str(x)[11:13]))
# basic_union["behavior_time"] = basic_union["behavior_time"].apply(lambda x:str(x)[:10])
# basic_union["behavior_time"] = pd.to_datetime(basic_union["behavior_time"])

##行为编码1 6-12, 12-17, 17-21, 21-05
def hour_rank(x):
    if x>=6 and x < 12:
        return 0
    elif x >=12 and x <17:
        return 1
    elif x >=17 and x <21:
        return 2
    elif x >=21 or x<6:
        return 3
basic_union["behavior_hour_rank"] = basic_union["behavior_hour"].apply(hour_rank)
def hour_type(df):
    if df["behavior_hour_rank"]==0 and df['behavior_type']==1:
        return 0
    elif df["behavior_hour_rank"]==0 and df['behavior_type']==2:
        return 1
    elif df["behavior_hour_rank"]==0 and df['behavior_type']==3:
        return 2
    elif df["behavior_hour_rank"]==1 and df['behavior_type']==1:
        return 3
    elif df["behavior_hour_rank"]==1 and df['behavior_type']==2:
        return 4
    elif df["behavior_hour_rank"]==1 and df['behavior_type']==3:
        return 5
    elif df["behavior_hour_rank"] == 2 and df['behavior_type'] == 1:
        return 6
    elif df["behavior_hour_rank"] == 2 and df['behavior_type'] == 2:
        return 7
    elif df["behavior_hour_rank"] == 2 and df['behavior_type'] == 3:
        return 8
    elif df["behavior_hour_rank"]==3 and df['behavior_type']==1:
        return 9
    elif df["behavior_hour_rank"]==3 and df['behavior_type']==2:
        return 10
    elif df["behavior_hour_rank"]==3 and df['behavior_type']==3:
        return 11
basic_union["hour_type"] = basic_union["behavior_hour_rank"]*3+basic_union['behavior_type']-1
#近7天的行为序列,最后一次行为距账单小时数及行为序列
basic_behavior_type = basic_union.loc[(basic_union["behavior_time"]>=basic_union["auditing_date_last7"])&(
                    basic_union["behavior_time"]<basic_union["auditing_date"])]
basic_behavior_type = basic_behavior_type.sort_values(by=["user_id","listing_id","behavior_time"])
basic_behavior_type["behavior_time_rank"] = basic_behavior_type.groupby(["user_id","listing_id"])["behavior_time"].rank(ascending=False,method='first')
basic_behavior_type["behavior_time_diff"] = (basic_behavior_type["auditing_date"]-basic_behavior_type["behavior_time"])
basic_behavior_type["behavior_time_diff"]=basic_behavior_type["behavior_time_diff"].dt.days*24+(basic_behavior_type["behavior_time_diff"].dt.seconds/3600).astype(int)
basic_behavior_type_union = basic_behavior_type.loc[basic_behavior_type["behavior_time_rank"]==1]
basic_behavior_type_union = basic_behavior_type_union[["user_id","listing_id","hour_type","behavior_time_diff"]]
basic_behavior_type_union.columns = ["user_id","listing_id","behavior_hour_type_last1","behavior_time_diff_last1"]
for i in [2,3,4,5,6,7,8,9,10]:
    temp = basic_behavior_type.loc[basic_behavior_type["behavior_time_rank"]==i][["user_id","listing_id","hour_type"]]
    temp.columns = ["user_id","listing_id","behavior_hour_type_last{}".format(i)]
    basic_behavior_type_union = basic_behavior_type_union.merge(temp,how='left',on=["user_id","listing_id"])
basic = basic.merge(basic_behavior_type_union,how='left',on=["user_id","listing_id"])
# # todo 聚合后用Embedding训练为矩阵
# from keras.models import Sequential
# from keras.layers import Embedding
# import numpy as np
# model = Sequential()
# model.add(Embedding(11, 1, input_length=3))
# model.compile('rmsprop', 'mse')
# input_array = [[1,5,9],[2,3,8],[2,3,8]]
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# print(output_array)

def daytime(df): #白天数量
    return df[(df >= 8) & (df < 22)].count()
def nighttime(df): #夜晚数量
    return df[(df < 8) | (df >= 22)].count()
def behavior1(df): #行为1/2/3
    return df[df == 1].count()
def behavior2(df):
    return df[df == 2].count()
def behavior3(df):
    return df[df == 3].count()
def hourrank0(df):
    return df[df==0].count()
def hourrank1(df):
    return df[df==1].count()
def hourrank2(df):
    return df[df==2].count()
def hourrank3(df):
    return df[df==3].count()
agg = {
    "behavior_hour": [daytime, nighttime, 'count'],
    "behavior_type": [behavior1, behavior2, behavior3],
    "behavior_hour_rank":[hourrank0,hourrank1,hourrank2,hourrank3]
}
for day in [1,3,7,15,30,60]:
    print(day)
    temp_union = basic_union.loc[(basic_union["behavior_time"]>=basic_union["auditing_date_last{}".format(day)])&(
                    basic_union["behavior_time"]<basic_union["auditing_date"])]

    basic_union_last = temp_union.groupby(["user_id", "listing_id"], as_index=False).agg(agg)
    basic_union_last.columns = ['behavior_last{}_'.format(day)+ i[0] + '_' + i[1] for i in basic_union_last.columns]
    basic_union_last = basic_union_last.rename(columns={'behavior_last{}_user_id_'.format(day):"user_id",'behavior_last{}_listing_id_'.format(day):"listing_id"})

    basic = basic.merge(basic_union_last,how='left',on=["user_id","listing_id"])

# 近1/7/15/30/45天行为数中1/2/3/白天/黑夜数占比 sum
for i in [1,7,15,30,60]:
    print("行为数占比：",i)
    basic["behavior_last{}_behavior1_ratio".format(i)] = basic["behavior_last{}_behavior_type_behavior1".format(i)]/basic["behavior_last{}_behavior_hour_count".format(i)]
    basic["behavior_last{}_behavior2_ratio".format(i)] = basic["behavior_last{}_behavior_type_behavior2".format(i)] / basic["behavior_last{}_behavior_hour_count".format(i)]
    basic["behavior_last{}_behavior3_ratio".format(i)] = basic["behavior_last{}_behavior_type_behavior3".format(i)] / basic["behavior_last{}_behavior_hour_count".format(i)]
    basic["behavior_last{}_behavior_hour_daytime_ratio".format(i)] = basic["behavior_last{}_behavior_hour_daytime".format(i)] / basic["behavior_last{}_behavior_hour_count".format(i)]
    basic["behavior_last{}_behavior_hour_nighttime_ratio".format(i)] = basic["behavior_last{}_behavior_hour_nighttime".format(i)] / basic["behavior_last{}_behavior_hour_count".format(i)]
# 近1天行为数中1/2/3/白天/黑夜数占7/15/30/45比
for i in [7,15,30,60]:
    print("近1天的占比：",i)
    basic["behavior_last1_behavior_hour_daytime_ratio_last{}".format(i)] = basic["behavior_last1_behavior_hour_daytime"]/basic["behavior_last{}_behavior_hour_daytime".format(i)]
    basic["behavior_last1_behavior_hour_nighttime_ratio_last{}".format(i)] = basic["behavior_last1_behavior_hour_nighttime"]/basic["behavior_last{}_behavior_hour_nighttime".format(i)]
    basic["behavior_last1_behavior_type_behavior1_ratio_last{}".format(i)] = basic["behavior_last1_behavior_type_behavior1"] / basic["behavior_last{}_behavior_type_behavior1".format(i)]
    basic["behavior_last1_behavior_type_behavior2_ratio_last{}".format(i)] = basic["behavior_last1_behavior_type_behavior2"] / basic["behavior_last{}_behavior_type_behavior2".format(i)]
    basic["behavior_last1_behavior_type_behavior3_ratio_last{}".format(i)] = basic["behavior_last1_behavior_type_behavior3"] / basic["behavior_last{}_behavior_type_behavior3".format(i)]

# 近7天行为数中1/2/3/白天/黑夜数占15/30/45比
for i in [15,30,60]:
    print("近7天的占比：",i)
    basic["behavior_last7_behavior_hour_daytime_ratio_last{}".format(i)] = basic["behavior_last7_behavior_hour_daytime"]/basic["behavior_last{}_behavior_hour_daytime".format(i)]
    basic["behavior_last7_behavior_hour_nighttime_ratio_last{}".format(i)] = basic["behavior_last7_behavior_hour_nighttime"]/basic["behavior_last{}_behavior_hour_nighttime".format(i)]
    basic["behavior_last7_behavior_type_behavior1_ratio_last{}".format(i)] = basic["behavior_last7_behavior_type_behavior1"] / basic["behavior_last{}_behavior_type_behavior1".format(i)]
    basic["behavior_last7_behavior_type_behavior2_ratio_last{}".format(i)] = basic["behavior_last7_behavior_type_behavior2"] / basic["behavior_last{}_behavior_type_behavior2".format(i)]
    basic["behavior_last7_behavior_type_behavior3_ratio_last{}".format(i)] = basic["behavior_last7_behavior_type_behavior3"] / basic["behavior_last{}_behavior_type_behavior3".format(i)]

# 近15天行为数中1/2/3/白天/黑夜数占30/45比
for i in [30,60]:
    print("近15天的占比：",i)
    basic["behavior_last15_behavior_hour_daytime_ratio_last{}".format(i)] = basic["behavior_last15_behavior_hour_daytime"]/basic["behavior_last{}_behavior_hour_daytime".format(i)]
    basic["behavior_last15_behavior_hour_nighttime_ratio_last{}".format(i)] = basic["behavior_last15_behavior_hour_nighttime"]/basic["behavior_last{}_behavior_hour_nighttime".format(i)]
    basic["behavior_last15_behavior_type_behavior1_ratio_last{}".format(i)] = basic["behavior_last15_behavior_type_behavior1"] / basic["behavior_last{}_behavior_type_behavior1".format(i)]
    basic["behavior_last15_behavior_type_behavior2_ratio_last{}".format(i)] = basic["behavior_last15_behavior_type_behavior2"] / basic["behavior_last{}_behavior_type_behavior2".format(i)]
    basic["behavior_last15_behavior_type_behavior3_ratio_last{}".format(i)] = basic["behavior_last15_behavior_type_behavior3"] / basic["behavior_last{}_behavior_type_behavior3".format(i)]

del basic["auditing_date_last1"]
del basic["auditing_date_last3"]
del basic["auditing_date_last7"]
del basic["auditing_date_last15"]
del basic["auditing_date_last30"]
del basic["auditing_date_last60"]
del basic["dead_line"]

basic.to_csv(outpath+'feature_behavior_logs_plus0619.csv',index=None)
