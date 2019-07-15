# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3model_pytorch.py
# @time  : 2019/6/24
"""
文件说明：pytorch 模型
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

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
y = "y_date_diff"
# y = "y_is_overdue"
n = 33 #分类数量，和y有关
#######################################
import sys
sys.path.append("/home/dev/lm/utils_lm")
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.nn import CrossEntropyLoss, Module,NLLLoss
from collections import Counter
from model_pytorch.fit_module import FitModule

from model_train.a1_preprocessing import NaFilterFeature #删除缺失过多的列
nf = NaFilterFeature(num=0.5)
X_train = nf.fit_transform(train[features])
X_test = nf.transform(test[features])

for i in X_train.columns:
    X_train[i] = X_train[i].fillna(X_train[i].median())
    X_test[i] = X_test[i].fillna(X_test[i].median())

y_train = train[y].values
X_train = X_train.values
X_test= X_test.values

n_feats = X_train.shape[1]
n_classes = 33

from sklearn.preprocessing import MinMaxScaler
minMax = MinMaxScaler() #归一化
X_train = minMax.fit_transform(X_train)
X_test = minMax.transform(X_test)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()

class ANN(FitModule): #.logloss 9.70
    def __init__(self, n_features, n_classes, hidden_size1=60,hidden_size2=40,hidden_size3=20):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, n_classes)


    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(self.dropout(x))
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(self.dropout(x))
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(self.dropout(x))
        x = self.fc4(x)
        x = F.softmax(x,dim=1)
        return x

model = ANN(n_feats, n_classes,hidden_size1=320,hidden_size2=240,hidden_size3=50)
# class MLP2(FitModule):
#     def __init__(self, n_feats, n_classes, nonlin=F.relu):
#         super(MLP2, self).__init__()
#
#         self.dense0 = nn.Linear(n_feats, 320)
#         self.nonlin = nonlin
#         self.dropout = nn.Dropout(0.5)
#         self.dense1 = nn.Linear(320, 120)
#         self.output = nn.Linear(120, n_classes)
#
#     def forward(self, X, **kwargs):
#         X = self.nonlin(self.dense0(X))
#         X = self.dropout(X)
#         X = F.relu(self.dense1(X))
#         X = F.softmax(self.output(X), dim=-1)
#         return X
# model = MLP2(n_feats, n_classes)
def accuracy(y_true, y_pred):
    return log_loss(y_true, y_pred)
# opt = partial(Adam, lr=0.001, betas=(0.9, 0.99)) #随机梯度下降
BATCH_SIZE = 40
RANDOM_STATE = 2
EPOCH = 5
SEED = 0
model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    verbose=1,
    validation_split=0.3,
    seed=SEED,
    metrics=[accuracy],
)

train_proba = model.predict(X_train, batch_size=BATCH_SIZE)
print(train_proba)
print("CV score: {:<8.5f}".format(log_loss(train[y], train_proba)))
test_proba = model.predict(X_test, batch_size=BATCH_SIZE)
print(test_proba)