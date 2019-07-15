# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3model_deepctr.py
# @time  : 2019/6/25
"""
文件说明：deepctr
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score,roc_auc_score

date = "0619"
oripath = "/home/dev/lm/paipai/ori_data/"
inpath = "/home/dev/lm/paipai/feature/"
outpath = "/home/dev/lm/paipai/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
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
df["y_date_diff_bin"] = df["y_date_diff_bin"].replace(-1,9)
df["y_date_diff_bin3"] = df["y_date_diff_bin3"].replace(-1,2)
df = df.replace([np.inf, -np.inf], np.nan) #正无穷负无穷均按照缺失处理

train = df[df["auditing_date"]<='2018-12-31']
train['repay_amt'] = train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train["y_date_diff"]=train["y_date_diff"].astype(int)
train["y_date_diff_bin"]=train["y_date_diff_bin"].astype(int)
train["y_date_diff_bin3"]=train["y_date_diff_bin3"].astype(int)
test = df[df["auditing_date"]>='2019-01-01']
print(train.shape)
print(test.shape)
# 字符变量处理

#无法入模的特征和y pred特征
del_feature = ["user_id","listing_id","auditing_date","due_date","repay_date","repay_amt"
                ,"user_info_tag_id_city","user_info_tag_taglist","dead_line",
               "other_tag_pred_is_overdue", "other_tag_pred_is_last_date",
               "user_info_tag_id_province", "user_info_tag_cell_province"]
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
# y = "y_date_diff"
y = "y_is_overdue"
n = 33 #分类数量，和y有关
#######################################
import sys
sys.path.append("/home/dev/lm/utils_lm")
import numpy as np

from model_train.a1_preprocessing import NaFilterFeature #删除缺失过多的列
nf = NaFilterFeature(num=0.5)
X_train = nf.fit_transform(train[features])
X_test = nf.transform(test[features])

for i in X_train.columns:
    X_train[i] = X_train[i].fillna(X_train[i].median())
    X_test[i] = X_test[i].fillna(X_test[i].median())

sparse_features  = [i for i in X_train.columns if (i[:6]=="other_" and len(i.split("_"))==2)]
sparse_features.extend(["auditing_month","basic_m_days","basic_day_of_week","basic_day_of_month"
                           ,"basic_day_of_week_due","basic_day_of_month_due","user_info_tag_is_c23141"
                           , "user_info_tag_is_c31255","user_info_tag_is_c20092","user_info_tag_is_c02321"
                           , "user_info_tag_is_N"])
dense_features = [i for i in X_train.columns if i not in sparse_features]
target = [y]
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM,DCN
from deepctr.utils import SingleFeat

#对稠密特征归一化
mms = MinMaxScaler(feature_range=(0, 1))
X_train[dense_features] = mms.fit_transform(X_train[dense_features])
X_test[dense_features] = mms.transform(X_test[dense_features])

#对稀疏特征编码
for feat in sparse_features:
    lbe = LabelEncoder()
    X_train[feat] = lbe.fit_transform(X_train[feat])
    X_test[feat] = lbe.transform(X_test[feat])

sparse_feature_list = [SingleFeat(feat, X_train[feat].nunique())  # since the input is string
                           for feat in sparse_features]
dense_feature_list = [SingleFeat(feat, 0, )
                            for feat in dense_features]

train_model_input = [X_train[feat.name].values for feat in sparse_feature_list] + \
                    [X_train[feat.name].values for feat in dense_feature_list]
test_model_input = [X_test[feat.name].values for feat in sparse_feature_list] + \
                   [X_test[feat.name].values for feat in dense_feature_list]

model = DeepFM({"sparse": sparse_feature_list,"dense": dense_feature_list}, task='binary')
model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )
history = model.fit(train_model_input, train[target].values,batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(train_model_input, batch_size=256)
print("train score", roc_auc_score(train[target].values, pred_ans)) #0.757
pred_ans = model.predict(test_model_input, batch_size=256)

ans = pd.DataFrame({"listing_id":test["listing_id"].values,"is_overdue":[i[0] for i in pred_ans]})
ans.to_csv(outpath+"is_overdue_dfm.csv",index=False)
