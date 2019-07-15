# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 1feature_select.py
# @time  : 2019/6/24
"""
文件说明：特征筛选
高缺失
共线性
低重要度
未使用
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

date = "300"
# oripath = "F:/数据集/1906拍拍/"
# inpath = "F:/数据集处理/1906拍拍/"
# outpath = "F:/项目相关/1906拍拍/out/"
oripath = "/home/dev/lm/paipai/ori_data/"
# inpath = "/home/dev/lm/paipai/feature/"
inpath = '/home/dev/lm/paipai/feature_ning/'
outpath = "/home/dev/lm/paipai/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic300.csv", encoding='utf8'))
print("feature_basic",df_basic.shape)
df_train = pd.read_csv(open(inpath + "feature_basic_train{}.csv".format(date), encoding='utf8'))
print("feature_basic_train",df_train.shape)
df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs{}.csv".format(date), encoding='utf8'))
print("feature_behavior_logs",df_behavior_logs.shape)
df_listing_info = pd.read_csv(open(inpath + "feature_listing_info{}.csv".format(date), encoding='utf8'))
print("feature_listing_info",df_listing_info.shape)
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs{}.csv".format(date), encoding='utf8'))
print("feature_repay_logs",df_repay_logs.shape)
df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag{}.csv".format(date), encoding='utf8'))
print("feature_user_info_tag",df_user_info_tag.shape)
df_other = pd.read_csv(open(inpath + "feature_other{}.csv".format(date), encoding='utf8'))
print("feature_other",df_other.shape)
#合并所有特征
df = df_basic.merge(df_train,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_behavior_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_listing_info,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_repay_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_user_info_tag,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_other,how='left',on=['user_id','listing_id','auditing_date'])
print(df.shape)
#调整多分类y
df["y_date_diff"] = df["y_date_diff"].replace(-1,32) #0~31


train = df[df["auditing_date"]<='2018-12-31']
train['repay_amt'] = train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train["y_date_diff"]=train["y_date_diff"].astype(int)
test = df[df["auditing_date"]>='2019-01-01']
print(train.shape)
print(test.shape)
# 字符变量处理

#无法入模的特征和y pred特征
del_feature = ["user_id","listing_id","auditing_date","due_date","repay_date","repay_amt"
                ,"user_info_tag_id_city","user_info_tag_taglist","dead_line",
               "other_tag_pred_is_overdue", "other_tag_pred_is_last_date",
               "user_info_tag_id_province", "user_info_tag_cell_province","is_train"]
y_list = [i  for i in df.columns if i[:2]=='y_']
del_feature.extend(y_list)
features = []
for col in df.columns:
    if col not in del_feature:
        features.append(col)
# catgory_feature = ["auditing_month","user_info_tag_gender","user_info_tag_cell_province","user_info_tag_id_province",
#                    "user_info_tag_is_province_equal"]
catgory_feature = ["auditing_month","user_info_tag_gender", "user_info_tag_is_province_equal"]
catgory_feature = [features.index(i) for i in catgory_feature]
y = "y_is_last_date"
n = 33 #分类数量，和y有关

import sys
sys.path.append("/home/dev/lm/utils_lm")

from model_train.a1_preprocessing import NaFilterFeature
nff = NaFilterFeature(num=0.8)
X_train = nff.fit_transform(train[features])
X_test = nff.transform(test[features])

from model_train.a2_feature_selection import coorelation,select_primaryvalue_ratio
features_spr, feature_primaryvalue_ratio = select_primaryvalue_ratio(X_train,ratiolimit=0.9)
features_coo, del_df = coorelation(X_train[features_spr],features_spr,coorelation_threshold=0.9)

#输出筛选后的特征
pd.DataFrame({"feature":features_coo}).to_csv(inpath+"feature300.csv",index=False)