# -*- coding: utf-8 -*-
#@author: limeng
#@file: 3model.py
#@time: 2019/6/7 16:59
"""
文件说明：整合特征建模,33分类
train 2018.1.1~2018.12.31
test 2019.2.1~2019.3.31
"""
import pandas as pd

inpath = "F:/数据集处理/1906拍拍/"
outpath = "F:/项目相关/1906拍拍/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
df_train = pd.read_csv(open(inpath + "feature_basic_train.csv", encoding='utf8'))
df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs.csv", encoding='utf8'))
df_listing_info = pd.read_csv(open(inpath + "feature_listing_info.csv", encoding='utf8'))
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs.csv", encoding='utf8'))
df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag.csv", encoding='utf8'))
#合并所有特征
df = df_basic.merge(df_train,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_behavior_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_listing_info,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_repay_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_user_info_tag,how='left',on=['user_id','listing_id','auditing_date'])
#调整多分类y
df["y_date_diff"] = df["y_date_diff"].replace(-1,32)
df["y_date_diff_bin"] = df["y_date_diff_bin"].replace(-1,9)
df["y_date_diff"] = df["y_date_diff"].replace(-1,2)
print(df.shape)
train = df[df["auditing_date"]<='2018-12-31']
test = df[df["auditing_date"]>='2019-01-01']
print(train.shape)
print(test.shape)
# 字符变量处理

#无法入模的特征和y
del_feature = ["user_id","listing_id","auditing_date","due_date","due_amt","repay_date","repay_amt"
               ,"due_date_d","repay_date_d","user_info_tag_id_city","user_info_tag_taglist"
            ,"y_date_diff","y_date_diff_bin","y_date_diff_bin3","y_is_overdue","y_is_last_date"]
features = []
for col in df.columns:
    if col not in del_feature:
        features.append(col)
catgory_feature = ["auditing_month","user_info_tag_gender","user_info_tag_cell_province","user_info_tag_id_province"]
y = "y_date_diff"
n = 33 #分类数量，和y有关
#开始训练
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np

param = {'objective': 'multiclass',
         'num_class':n,
         'num_leaves': 31,
         'min_data_in_leaf': 25,
         'max_depth': 5,  # 7
         'learning_rate': 0.01,
         'lambda_l1': 0.13,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         'bagging_freq': 8,
         "bagging_fraction": 0.9,
         "metric": 'multi_logloss',
         "verbosity": -1,
         "random_state": 2333}

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
folds = KFold(n_splits=3, shuffle=True, random_state=2333)
oof = np.zeros((len(train),n))
predictions = np.zeros((len(test),n))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], train[y].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=train[y].iloc[trn_idx],
                           categorical_feature=catgory_feature)
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=train[y].iloc[val_idx],
                           categorical_feature=catgory_feature)

    num_round = 2000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=10,categorical_feature=catgory_feature)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(train[y], oof)))
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)

#train
train_prob = pd.DataFrame(oof)
train_dic = {
    "user_id": train["user_id"].values,
    "listing_id":train["listing_id"].values,
    "auditing_date":train["auditing_date"].values,
    "due_date":train["due_date"].values,
    "due_amt":train["due_amt"].values,
}
for key in train_dic:
    train_prob[key] = train_dic[key]
#test
test_prob = pd.DataFrame(predictions)
test_dic = {
    "user_id": test["user_id"].values,
    "listing_id":test["listing_id"].values,
    "auditing_date":test["auditing_date"].values,
    "due_date":test["due_date"].values,
    "due_amt":test["due_amt"].values,
}
for key in test_dic:
    test_prob[key] = test_dic[key]
for i in range(n-1):
    test_prob[i] = test_prob[i]*test_prob["due_amt"]
#评价

#提交
def df_rank(df_prob, df_sub):
    for i in range(33):
        df_tmp = df_prob[['listing_id', i]]
        df_tmp['rank'] = i+1
        df_sub = df_sub.merge(df_tmp,how='left',on=["listing_id",'rank'])
        df_sub.loc[df_sub['rank']==i+1,'repay_amt']=df_sub.loc[df_sub['rank']==i+1,i]
    return df_sub[['listing_id','repay_amt','repay_date']]
submission = pd.read_csv(open("F:/数据集/1906拍拍/"+"submission.csv",encoding='utf8'))
submission['repay_date'] = pd.to_datetime(submission['repay_date'])
submission['rank'] = submission.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
sub = df_rank(test_prob, submission)
sub.to_csv(outpath+'sub_lgb_33_0609.csv',index=None)

# #train mse文件构建,太耗内存无法运行
# from dateutil.relativedelta import relativedelta
# date_lst = []
# listing_lst = []
# n = 0
# for id in train["listing_id"]:
#     n = n+1
#     print(n)
#     temp_lst = [id]*365
#     listing_lst.extend(temp_lst)
# train_date = pd.DataFrame({"listing_id":listing_lst,"repay_date":date_lst})
# train["auditing_date"] = pd.to_datetime(train["auditing_date"])
# train["due_date"] = pd.to_datetime(train["due_date"])
# train["repay_date"] = pd.to_datetime(train["repay_date"])
# train_date = train_date.merge(train[["listing_id","auditing_date","due_date"]],how='left',on="listing_id")
# train_date = train_date.loc[(train_date['repay_date']>=train_date['auditing_date'])&(train_date['repay_date']<=train_date['due_date'])]
# train_date = train_date.merge(train[["listing_id","repay_date","repay_amt"]],how='left',on=["listing_id","repay_date"])
# train_date = train_date.dillna(0)
# train_date.to_csv(inpath+'submission_train.csv',index=None)