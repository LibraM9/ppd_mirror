# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 4rule_tag.py
# @time  : 2019/6/14
"""
文件说明：对tag进行处理,建模或者做规则
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

# oripath = "F:/数据集/1906拍拍/"
# inpath = "F:/数据集处理/1906拍拍/"
# outpath = "F:/项目相关/1906拍拍/out/"
oripath = "/home/dev/lm/paipai/ori_data/"
inpath = "/home/dev/lm/paipai/feature/"
outpath = "/home/dev/lm/paipai/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
print(df_basic.shape)
df_train = pd.read_csv(open(inpath + "feature_basic_train.csv", encoding='utf8'))
print(df_train.shape)
df_behavior_logs = pd.read_csv(open(inpath + "feature_behavior_logs.csv", encoding='utf8'))
print(df_behavior_logs.shape)
df_listing_info = pd.read_csv(open(inpath + "feature_listing_info.csv", encoding='utf8'))
print(df_listing_info.shape)
df_repay_logs = pd.read_csv(open(inpath + "feature_repay_logs.csv", encoding='utf8'))
print(df_repay_logs.shape)
df_user_info_tag = pd.read_csv(open(inpath + "feature_user_info_tag.csv", encoding='utf8'))
print(df_user_info_tag.shape)
df_other = pd.read_csv(open(inpath + "feature_other.csv", encoding='utf8'))
print(df_other.shape)
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

train = df[df["auditing_date"]<='2018-12-31']
train['repay_amt'] = train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float32')
train["y_date_diff"]=train["y_date_diff"].astype(int)
train["y_date_diff_bin"]=train["y_date_diff_bin"].astype(int)
train["y_date_diff_bin3"]=train["y_date_diff_bin3"].astype(int)
test = df[df["auditing_date"]>='2019-01-01']
print(train.shape)
print(test.shape)

features = ["listing_id",'user_info_tag_taglist',"user_info_tag_id_city"]
y_list = [i  for i in df.columns if i[:2]=='y_']
features.extend(y_list)
del_deatures = y_list.append('listing_id')

train = train[features]

#todo 人物画像
"""
最后还款 2485 2869 58 1234/2785 2168 2648 1915 2869 3156 3043 3425（0.05）3843 4332 3933 4853 5443
554 5545 735 58
是否逾期 3574 /1654 1804 1911 2710 2017(0.003) 3574 3043（0.003）421 4185 5628
"""
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

train['user_info_tag_taglist'] = train['user_info_tag_taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
cv = CountVectorizer(min_df=10, max_df=0.9)
train_cv = cv.fit_transform(train['user_info_tag_taglist'])

train_x = pd.DataFrame()
train_x = sparse.hstack((train_x.values, train_cv), format='csr', dtype='float32')
# train_x = pd.DataFrame(train_cv.toarray()) #转化为df无法跑模型，内存消耗过大
test['user_info_tag_taglist'] = test['user_info_tag_taglist'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
test_x = pd.DataFrame(cv.transform(test['user_info_tag_taglist']).toarray())

name = cv.vocabulary_
name = sorted(name.items(), key=lambda x: x[1], reverse=False)
name = [i[0] for i in name]
train_x.columns = name
test_x.columns = name
features = name
#####IV测试
train_x = train_x.iloc[:,5000:]
train_x = train_x.astype(str)
# train_x = train_x[["2785","2869","58","1234"]]
train_x["y_is_last_date"] = train["y_is_last_date"]
train_x["y_is_overdue"] = train["y_is_overdue"]
import sys
sys.path.append("/home/dev/lm/utils_lm")
from model_train.a2_feature_selection import iv
iv_out = iv(train_x.drop("y_is_overdue",axis=1),train_x["y_is_overdue"])[1]
print(iv_out)
#####下载高IV数据
feature_tag = ['2485', '2869', '58', '1234', '2785', '2168', '2648', '1915', '3156', '3043',
               '3425','3843', '4332', '3933' ,'4853', '5443','554', '5545', '735','3574',
              '1654', '1804', '1911', '2710', '2017','421', '4185', '5628']
train_x = train_x[feature_tag]
train_x["listing_id"] = train["listing_id"].values
test_x = test_x[feature_tag]
test_x["listing_id"] = test["listing_id"].values
out = pd.concat([train_x,test_x],axis=0)
out.to_csv(inpath+"user_tag_iv.csv",index=False)
#####训练
# y = "y_is_last_date"  #auc 0.57704
y = "y_is_overdue" # auc 0.55518
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
import numpy as np

param ={'num_leaves': 2**5,
         'min_data_in_leaf': 32,
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.03,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.5,
          "verbosity": -1,
        'is_unbalance': True
        }
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
# folds = KFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train[y].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],
                           label=train[y][trn_idx],)
    val_data = lgb.Dataset(train_x[val_idx],
                           label=train[y][val_idx],)

    num_round = 5000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,
                    early_stopping_rounds=100)

    oof[val_idx] = clf.predict(train_x[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(train[y].values, oof)))
print("auc score:",roc_auc_score(train[y].values, oof))
feature_importance = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
ascending=False)

out_train = pd.DataFrame(train["listing_id"])
out_train["pred"+y] = oof
out_test = pd.DataFrame(test["listing_id"])
out_test["pred"+y] = predictions


out = pd.concat([out_train,out_test],axis=0)
out.to_csv(inpath+"user_tag_pred.csv",index=False)