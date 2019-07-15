# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 3model_33_regression.py
# @time  : 2019/7/4
"""
文件说明：回归
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

date = "0619"
# oripath = "F:/数据集/1906拍拍/"
# inpath = "F:/数据集处理/1906拍拍/"
# outpath = "F:/项目相关/1906拍拍/out/"
oripath = "/home/dev/lm/paipai/ori_data/"
# inpath = "/home/dev/lm/paipai/feature/"
inpath = "/home/dev/lm/paipai/feature_ning/"
outpath = "/home/dev/lm/paipai/out/"
# 100特征
# df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
# print("feature_basic",df_basic.shape)
# df_train = pd.read_csv(open(inpath + "feature_basic_train{}.csv".format(date), encoding='utf8'))
# print("feature_basic_train",df_train.shape)
# df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs_plus{}.csv".format(date), encoding='utf8'))
# print("feature_behavior_logs",df_behavior_logs.shape)
# df_listing_info = pd.read_csv(open(inpath + "feature_listing_info{}.csv".format(date), encoding='utf8'))
# print("feature_listing_info",df_listing_info.shape)
# # df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs{}.csv".format(date), encoding='utf8'))
# # print("feature_repay_logs",df_repay_logs.shape)
# df_repay_logs2 = pd.read_csv(open(inpath + "feature_repay_logs_plus0702.csv", encoding='utf8'))
# print("feature_repay_logs2",df_repay_logs2.shape)
# df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag{}.csv".format(date), encoding='utf8'))
# print("feature_user_info_tag",df_user_info_tag.shape)
# df_other = pd.read_csv(open(inpath + "feature_other{}.csv".format(date), encoding='utf8'))
# print("feature_other",df_other.shape)

#300W特征
df_basic = pd.read_csv(open(inpath + "feature_basic300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_basic",df_basic.shape)
df_train = pd.read_csv(open(inpath + "feature_basic_train300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_basic_train",df_train.shape)
df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_behavior_logs",df_behavior_logs.shape)
df_listing_info = pd.read_csv(open(inpath + "feature_listing_info300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_listing_info",df_listing_info.shape)
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_repay_logs",df_repay_logs.shape)
df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_user_info_tag",df_user_info_tag.shape)
df_other = pd.read_csv(open(inpath + "feature_other300.csv", encoding='utf8'),parse_dates=['auditing_date'])
print("feature_other",df_other.shape)

#合并所有特征
df = df_basic.merge(df_train,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_behavior_logs,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_listing_info,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_repay_logs,how='left',on=['user_id','listing_id','auditing_date'])
# df = df.merge(df_repay_logs2,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_user_info_tag,how='left',on=['user_id','listing_id','auditing_date'])
df = df.merge(df_other,how='left',on=['user_id','listing_id','auditing_date'])
print(df.shape)
#调整多分类y
# df["y_date_diff"] = df["y_date_diff"].replace(-1,32) #0~31
# df["y_date_diff_bin"] = df["y_date_diff_bin"].replace(-1,9)
# df["y_date_diff_bin3"] = df["y_date_diff_bin3"].replace(-1,2)
df = df.replace([np.inf, -np.inf], np.nan)

# train = df[df["auditing_date"]<='2018-12-31']
train = df[df["is_train"]==1]
train = train[train["auditing_date"]<"2019-01-01"]
train['repay_amt'] = train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train["y_date_diff"]=train["y_date_diff"].astype(int)
# train["y_date_diff_bin"]=train["y_date_diff_bin"].astype(int)
# train["y_date_diff_bin3"]=train["y_date_diff_bin3"].astype(int)
# test = df[df["auditing_date"]>='2019-01-01']
test = df[df["is_train"]==0]
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
#读取筛选后的特征
# features = pd.read_csv(open(inpath + "feature.csv", encoding='utf8'))
# features = features["feature"].values.tolist()
# catgory_feature = ["auditing_month","user_info_tag_gender","user_info_tag_cell_province","user_info_tag_id_province",
#                    "user_info_tag_is_province_equal"]
catgory_feature = ["auditing_month","user_info_tag_gender", "user_info_tag_is_province_equal"]
catgory_feature = [features.index(i) for i in catgory_feature if i in features]
y = "y_date_diff"

# train = train.loc[train[y]!=32] #删除逾期数据

#开始训练 lgb  ######################################
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np

param = {'num_leaves': 31,
         'min_data_in_leaf': 32,
         'objective':'regression',
         'max_depth': 5,
         'learning_rate': 0.08,
         # "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 50,
        "verbosity": -1,
         "random_state": 2333,}

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
folds = KFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

train_x = train[features].values
test_x = test[features].values

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train[y].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],
                           label=train[y].iloc[trn_idx],
                           categorical_feature=catgory_feature)
    val_data = lgb.Dataset(train_x[val_idx],
                           label=train[y].iloc[val_idx],
                           categorical_feature=catgory_feature)

    num_round = 2000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,
                    early_stopping_rounds=20,categorical_feature=catgory_feature)
    #n*33矩阵
    val_pred_prob_everyday = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)
    oof[val_idx] = val_pred_prob_everyday

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(train[y], oof)))
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)

feature_importance.to_csv(outpath+"importance_reg_lgb300_623.csv")

#test
test_prob = pd.DataFrame(np.zeros((len(test),33)))
test_prob["reg"] = predictions.copy()
test_prob["reg"] = test_prob["reg"].apply(lambda x: -1 if x <-1 else x)
test_prob["reg"] = test_prob["reg"].apply(lambda x: 31 if x >31 else x)
def trans(df):
    """转换，回归结果转化为概率
    :param df:
    :return:
    """
    n = df["reg"]
    if n == -1:
        df[32]=1
    elif n == 31:
        df[31]=1
    elif n < 0:
        df[32] = -n
        df[0] = 1+n
    else:
        df[int(n)]=int(n)+1-n
        df[int(n)+1] = n-int(n)
    return df
test_prob = test_prob.apply(trans,axis=1)
test_dic = {
    "user_id": test["user_id"].values,
    "listing_id":test["listing_id"].values,
    "auditing_date":test["auditing_date"].values,
    "due_date":test["due_date"].values,
    "due_amt":test["due_amt"].values,
}
for key in test_dic:
    test_prob[key] = test_dic[key]
#输出预测概率
# test_prob.to_csv(outpath+'out_lgb_test.csv',index=None)
n=33
for i in range(n-1):
    test_prob[i] = test_prob[i]*test_prob["due_amt"]
#对于训练集评价
def df_rank(df_prob, df_sub):
    for i in range(33):
        print('转换中',i)
        df_tmp = df_prob[['listing_id', i]]
        df_tmp['rank'] = i+1
        df_sub = df_sub.merge(df_tmp,how='left',on=["listing_id",'rank'])
        df_sub.loc[df_sub['rank']==i+1,'repay_amt']=df_sub.loc[df_sub['rank']==i+1,i]
    return df_sub[['listing_id','repay_amt','repay_date']]
#
# submission_train =pd.read_csv(open(inpath+"submission_train.csv",encoding='utf8'),parse_dates=["repay_date"])
# submission_train['rank'] = submission_train.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
# sub_train = df_rank(train_prob, submission_train)
# sub_train = sub_train.merge(submission_train,how='left',on=['listing_id','repay_date'])
# print('mse_real:', mean_squared_error(sub_train['repay_amt_x'], sub_train['repay_amt_y']))
#提交
submission = pd.read_csv(open(oripath+"submission.csv",encoding='utf8'),parse_dates=["repay_date"])
submission['rank'] = submission.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
sub = df_rank(test_prob, submission)
sub.to_csv(outpath+'sub_reg_300_623.csv',index=None)