# -*- coding: utf-8 -*-
# @Author: limeng
# @File  : 4stacking.py
# @time  : 2019/6/28
"""
文件说明：模型stacking
"""
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score

date = "0619"
# oripath = "F:/数据集/1906拍拍/"
# inpath = "F:/数据集处理/1906拍拍/"
# outpath = "F:/项目相关/1906拍拍/out/"
oripath = "/home/dev/lm/paipai/ori_data/"
inpath = "/home/dev/lm/paipai/feature/"
outpath = "/home/dev/lm/paipai/out/"

df_basic = pd.read_csv(open(inpath + "feature_basic.csv", encoding='utf8'))
#调整多分类y
df_basic["y_date_diff"] = df_basic["y_date_diff"].replace(-1,32) #0~31
df_basic["y_date_diff_bin"] = df_basic["y_date_diff_bin"].replace(-1,9)
df_basic["y_date_diff_bin3"] = df_basic["y_date_diff_bin3"].replace(-1,2)
train1 = pd.read_csv(open(outpath + "out_lgb_train.csv", encoding='utf8'))
train1.columns = ["lgb"+str(i) for i in range(33)]+["user_id","listing_id","auditing_date","due_date","due_amt"]
train2 = pd.read_csv(open(outpath + "out_lgb368_train.csv", encoding='utf8'))
train2.columns = ["lgb368"+str(i) for i in range(33)]+["user_id","listing_id","auditing_date","due_date","due_amt"]
train3 = pd.read_csv(open(outpath + "out_dfm368_train.csv", encoding='utf8'))
train3.columns = ["dfm"+str(i) for i in range(33)]+["user_id","listing_id","auditing_date","due_date","due_amt"]
# train4 = pd.read_csv(open(outpath + "out_dfm368_node_train.csv", encoding='utf8'))
# train4.columns = ["dfm368"+str(i) for i in range(33)]+["user_id","listing_id","auditing_date","due_date","due_amt"]
test1 = pd.read_csv(open(outpath + "out_lgb_test.csv", encoding='utf8'))
test1.columns = train1.columns
test2 = pd.read_csv(open(outpath + "out_lgb368_test.csv", encoding='utf8'))
test2.columns = train2.columns
test3 = pd.read_csv(open(outpath + "out_dfm368_test.csv", encoding='utf8'))
test3.columns = train3.columns
# test4 = pd.read_csv(open(outpath + "out_dfm368_node_test.csv", encoding='utf8'))
# test4.columns = train4.columns

# train = pd.concat([train1,train2[train2.columns[:33]],train3[train3.columns[:33]],train4[train4.columns[:33]]],axis=1)
# test = pd.concat([test1,test2[train2.columns[:33]],test3[train3.columns[:33]],test4[train4.columns[:33]]],axis=1)
train = pd.concat([train1,train2[train2.columns[:33]],train3[train3.columns[:33]]],axis=1)
test = pd.concat([test1,test2[train2.columns[:33]],test3[train3.columns[:33]]],axis=1)

train = train.merge(df_basic,how='left',on=["user_id","listing_id"])
test = test.merge(df_basic,how='left',on=["user_id","listing_id"])

#无法入模的特征和y pred特征
del_feature = ["user_id","listing_id","auditing_date","due_date","repay_date","repay_amt"
                ,"user_info_tag_id_city","user_info_tag_taglist","dead_line",
               "other_tag_pred_is_overdue", "other_tag_pred_is_last_date",
               "user_info_tag_id_province", "user_info_tag_cell_province",
                'auditing_date_x', 'due_date_x', 'due_amt_x',
                'auditing_date_y', 'due_date_y', 'due_amt_y']
y_list = [i  for i in train.columns if i[:2]=='y_']
del_feature.extend(y_list)
features = []
for col in train.columns:
    if col not in del_feature:
        features.append(col)

y = "y_date_diff"
n = 33 #分类数量，和y有关

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import lightgbm as lgb
import numpy as np

param = {'objective': 'multiclass',
         'num_class':n,
         'num_leaves': 2**5, #2**5
         'min_data_in_leaf': 25,#25
         'max_depth': 5,  # 5 2.02949 4 2.02981
         'learning_rate': 0.05, # 0.02 0.04 2.05467 7318
         'lambda_l1': 0.13,
         "boosting": "gbdt",
         "feature_fraction": 0.85,
         'bagging_freq': 8,
         "bagging_fraction": 0.9, #0.9
         "metric": 'multi_logloss',
         "verbosity": -1,
         "random_state": 2333}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
# folds = KFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros((len(train),n))
predictions = np.zeros((len(test),n))
feature_importance_df = pd.DataFrame()

train_x = train[features].values.astype(np.float)
test_x = test[features].values.astype(np.float)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train[y].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(train_x[trn_idx],
                           label=train[y].iloc[trn_idx],)
    val_data = lgb.Dataset(train_x[val_idx],
                           label=train[y].iloc[val_idx],)

    num_round = 2000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=50,
                    early_stopping_rounds=20)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test_x, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(train[y], oof)))
feature_importance = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean().sort_values(by="importance",
ascending=False)

#test
test_prob = pd.DataFrame(predictions)
test_dic = {
    "user_id": test["user_id"].values,
    "listing_id":test["listing_id"].values,
    "auditing_date":test["auditing_date_x"].values,
    "due_date":test["due_date_x"].values,
    "due_amt":test["due_amt_x"].values,
}
for key in test_dic:
    test_prob[key] = test_dic[key]
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

#提交
submission = pd.read_csv(open(oripath+"submission.csv",encoding='utf8'),parse_dates=["repay_date"])
submission['rank'] = submission.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
sub = df_rank(test_prob, submission)
sub.to_csv(outpath+'sub_stack_model3_md5.csv',index=None)
