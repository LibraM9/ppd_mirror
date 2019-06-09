# -*- coding: utf-8 -*-
#@author: limeng
#@file: 2feature_user_info_tag.py
#@time: 2019/6/8 13:16
"""
文件说明：对于user_info user_taglist的处理
"""
import pandas as pd
import numpy as np
import gc

path = "F:/数据集/1906拍拍/"
outpath = "F:/数据集处理/1906拍拍/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'))

user_info = pd.read_csv(open(path+"user_info.csv",encoding='utf8'))
user_taglist = pd.read_csv(open(path+"user_taglist.csv",encoding='utf8'))

#user_info
basic = basic.merge(user_info,how='left',on='user_id')
basic["auditing_date"] = pd.to_datetime(basic["auditing_date"])
basic["insertdate"] = pd.to_datetime(basic["insertdate"])
basic = basic.loc[basic["insertdate"]<basic["auditing_date"]]
basic = basic.sort_values(["user_id", "listing_id","insertdate"])
basic["rank"] = basic.groupby(["user_id","listing_id"])["insertdate"].rank(ascending=False)
basic = basic.loc[basic["rank"]==1]
#1个用户可能有多条数据，需要卡时间
user_taglist["taglist"] = user_taglist["taglist"].apply(lambda x:x.split("|"))
user_taglist["tag_cnt"] = user_taglist["taglist"].apply(lambda x:len(x))
user_taglist["insertdate"] = pd.to_datetime(user_taglist["insertdate"])

basic = basic.merge(user_taglist,how='left',on=['user_id',"insertdate"])

#对特征进行处理
basic["reg_mon"] = pd.to_datetime(basic["reg_mon"])
basic["reg_year"] = basic["reg_mon"].apply(lambda x:int(str(x)[:4]))
basic["reg_month"] = basic["reg_mon"].apply(lambda x:int(str(x)[5:7]))
basic["month_diff"] = (basic["auditing_date"] -basic["reg_mon"]).apply(lambda x:int(x.days/30))
basic['gender'] = basic['gender'].apply(lambda x:1 if x== '男' else 0)
basic['is_province_equal'] = basic.apply(lambda x:1 if x["cell_province"]==x["id_province"] else 0,axis=1)

def trans_province(x):
    try:
        return int(x[1:])
    except:
        return np.nan
basic['cell_province'] = basic['cell_province'].apply(lambda x:trans_province(x))
basic['id_province'] = basic['id_province'].apply(lambda x:trans_province(x))

del basic["reg_mon"]
del basic["insertdate"]
del basic["rank"]
dic = {}
for i in basic.columns:
    if i not in ["user_id","listing_id","auditing_date"]:
        dic[i]="user_info_tag_"+i
basic = basic.rename(columns=dic)
basic.to_csv(outpath+'feature_user_info_tag.csv',index=None)