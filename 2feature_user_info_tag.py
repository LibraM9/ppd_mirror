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

# path = "F:/数据集/1906拍拍/"
# outpath = "F:/数据集处理/1906拍拍/"
path = "/data/dev/lm/paipai/ori_data/"
outpath = "/data/dev/lm/paipai/feature/"
# Y指标基础表
basic = pd.read_csv(open(outpath + "feature_main_key.csv", encoding='utf8'),parse_dates=['auditing_date'])

user_info = pd.read_csv(open(path+"user_info.csv",encoding='utf8'),parse_dates=['insertdate'])
user_taglist = pd.read_csv(open(path+"user_taglist.csv",encoding='utf8'),parse_dates=['insertdate'])

#user_info
basic = basic.merge(user_info,how='left',on='user_id')
basic = basic.loc[basic["insertdate"]<basic["auditing_date"]]
basic = basic.sort_values(["user_id", "listing_id","insertdate"],ascending=[True,True,False])
# basic["rank"] = basic.groupby(["user_id","listing_id"])["insertdate"].rank(ascending=False)
# basic = basic.loc[basic["rank"]==1]
basic = basic.drop_duplicates(['user_id',"listing_id"]).reset_index(drop=True) #去重，保留出现的第条数据
#1个用户可能有多条数据，需要卡时间

user_taglist["tag_cnt"] = user_taglist["taglist"].apply(lambda x:x.split("|")).apply(lambda x:len(x))

basic = basic.merge(user_taglist,how='left',on=['user_id',"insertdate"])

#对特征进行处理
basic["reg_mon"] = pd.to_datetime(basic["reg_mon"])
basic["reg_year"] = basic["reg_mon"].apply(lambda x:int(str(x)[:4]))
basic["reg_month"] = basic["reg_mon"].apply(lambda x:int(str(x)[5:7]))
basic["month_diff"] = (basic["auditing_date"] -basic["reg_mon"]).apply(lambda x:int(x.days/30))
basic['gender'] = basic['gender'].apply(lambda x:1 if x== '男' else 0)
def is_equal(x):
    if x["cell_province"]==x["id_province"]:
        return 1
    elif x["cell_province"] == "\\N":
        return np.nan
    else:
        return 0
basic['is_province_equal'] = basic.apply(is_equal,axis=1)

def trans_province(x):
    try:
        return int(x[1:])
    except:
        return np.nan
basic['cell_province'] = basic['cell_province'].apply(lambda x:trans_province(x))
basic['id_province'] = basic['id_province'].apply(lambda x:trans_province(x))

basic["is_c23141"] = basic["id_city"].apply(lambda x:1 if x=="c23141" else 0)
basic["is_c31255"] = basic["id_city"].apply(lambda x:1 if x=="c31255" else 0)
basic["is_c20092"] = basic["id_city"].apply(lambda x:1 if x=="c20092" else 0)
basic["is_c02321"] = basic["id_city"].apply(lambda x:1 if x=="c02321" else 0)
basic["is_N"] = basic["id_city"].apply(lambda x:1 if x=="\\N" else 0)

del basic["reg_mon"]
del basic["insertdate"]
dic = {}
for i in basic.columns:
    if i not in ["user_id","listing_id","auditing_date"]:
        dic[i]="user_info_tag_"+i
basic = basic.rename(columns=dic)
basic.to_csv(outpath+'feature_user_info_tag0619.csv',index=None)