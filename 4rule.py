# -*- coding: utf-8 -*-
#@author: li
#@file: rule.py
#@time: 2019/6/7 16:58
"""
文件说明：规则构建

1. model_33_no032 对于1~31的效果远好于 model_33 +600分
2. 全阈值错误较多,-500分
3. 默认阈值model_33+默认阈值以外model_33_no032可提高600分左右
4. 对0 32 的覆盖auc*0.5为最优
"""
import pandas as pd

oripath = "F:/数据集/1906拍拍/"
inpath = "F:/数据集处理/1906拍拍/"
outpath = "F:/项目相关/1906paipai/sub/"

test = pd.read_csv(open(oripath+"test.csv",encoding='utf8')) #
model_33 = pd.read_csv(open(outpath+"sub_lgb_33_0613.csv",encoding='utf8'),parse_dates=["repay_date"]) #
model_33 = model_33.merge(test[["listing_id","due_amt"]],how='left',on='listing_id')
model_33_no032 = pd.read_csv(open(outpath+"lgb_33_no032_0613.csv",encoding='utf8'),parse_dates=["repay_date"]) #
model_33_no032 = model_33_no032.merge(test[["listing_id","due_amt"]],how='left',on='listing_id')

model_33['rank'] = model_33.groupby(['listing_id'])['repay_date'].rank(ascending=False,method='first')
model_33_no032['rank'] = model_33_no032.groupby(['listing_id'])['repay_date'].rank(ascending=False,method='first')

#以model_33为基础覆盖一定为0的和一定为32的
dic = {0:0.408187,
1:0.121085,
2:0.05943,
3:0.056404,
4:0.026425,
5:0.02138,
6:0.017568,
7:0.014797,
8:0.012993,
9:0.011393,
10:0.009984,
11:0.009002,
12:0.008219,
13:0.007688,
14:0.00692,
15:0.006443,
16:0.006231,
17:0.005832,
18:0.005492,
19:0.005108,
20:0.004788,
21:0.004504,
22:0.004295,
23:0.004197,
24:0.003922,
25:0.003934,
26:0.00393,
27:0.004102,
28:0.004677,
29:0.005645,
30:0.009865,
31:0.008368,
32:0.117192}

def cover(model_33, is_nums, threshold=None):
    """
    :param model:
    :param is_num:
    :param threshold: 人数占比
    :return:
    """

    model = model_33.copy()
    is_num = is_nums.copy()
    class_ratio = is_num.columns[-1]
    if class_ratio=="last_date":
        c = 0
    elif class_ratio=="overdue":
        c = 32
    else:
        c = class_ratio.split("_")[-1]
    print("对{}进行覆盖".format(c))
    is_num["rank"] = is_num[class_ratio].rank(ascending=False,method='first')
    if threshold==None:
        threshold = int(is_num.shape[0]*dic[c]) #测试集中出现某一类的可能数量
    else:
        print(threshold)
        threshold = int(is_num.shape[0]*threshold)
    print("覆盖的样本量:",threshold)
    print("分类阈值:", is_num[is_num["rank"]==threshold][class_ratio].values)
    cover_id = set(is_num[is_num["rank"]<=threshold]["listing_id"]) #提取符合条件的id

    if c == 32:
        model.loc[model['listing_id'].isin(cover_id) == True, "repay_amt"]=0
    else:
        model.loc[model['listing_id'].isin(cover_id) == True, "repay_amt"] = 0
        model.loc[(model['listing_id'].isin(cover_id) == True) & (model['rank'] == c + 1), "repay_amt"] = \
            model.loc[(model['listing_id'].isin(cover_id) == True) & (model['rank'] == c + 1)]["due_amt"]
    return model

# auc 信息
auc = pd.read_csv(open(outpath+"auc_model1_31.csv",encoding='utf8'))

# #对0覆盖
# is_last_date = pd.read_csv(open(outpath+"is_last_date0613.csv",encoding='utf8')) #
# model_33_no032 = cover(model_33_no032,is_last_date)
# #对32覆盖
# is_overdue = pd.read_csv(open(outpath+"is_overdue0613.csv",encoding='utf8')) #
# model_33_no032 = cover(model_33_no032,is_overdue)
#
# model_33_no032[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_lgbno032_cover032_0614.csv",index=None)

#对0覆盖.auc*0.5为最优 6404
n=0
is_last_date = pd.read_csv(open(outpath+"is_last_date0613.csv",encoding='utf8')) #
model_33 = cover(model_33,is_last_date,threshold=0.3*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# model_33 = cover(model_33,is_last_date)
#对32覆盖.auc*0.5为最优
n=32
is_overdue = pd.read_csv(open(outpath+"is_overdue0613.csv",encoding='utf8')) #
model_33 = cover(model_33,is_overdue,threshold=0.3*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# model_33 = cover(model_33,is_overdue)

model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_lgb_cover032_auc30_0614.csv",index=None)


#拼接model_33_no032和model_33，对最有可能为1~31的数据替换为model_33_no032
#1. 0.408187 last_date 和 0.117192 overdue id（获得最可能为0和32的ID）
#2. 以覆盖了auc阈值的model_33为基础，其余ID均认为是1~31的ID 拼入 model_33_no032
last_overdue_id = []
#0
is_last_date["rank"] = is_last_date["last_date"].rank(ascending=False,method='first')
threshold = int(is_last_date.shape[0]*(dic[0]+0.1))
cover_id = list(set(is_last_date[is_last_date["rank"]<=threshold]["listing_id"]))
last_overdue_id.extend(list(cover_id))
#32
is_overdue["rank"] = is_overdue["overdue"].rank(ascending=False,method='first')
threshold = int(is_overdue.shape[0]*(dic[32]+0.05))
cover_id = list(set(is_overdue[is_overdue["rank"]<=threshold]["listing_id"]))
last_overdue_id.extend(list(cover_id))
last_overdue_id = set(last_overdue_id) #76477/默认阈值63700
model_33.loc[model_33["listing_id"].isin(last_overdue_id)==False,'repay_amt']= \
    model_33_no032.loc[model_33_no032["listing_id"].isin(last_overdue_id)==False]['repay_amt']
model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover032_auc30_0614.csv",index=None)

# todo 对1~31进行覆盖