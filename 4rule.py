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

规则：overdue不减去全部值，减去部分
"""
import pandas as pd

oripath = "F:/数据集/1906拍拍/"
inpath = "F:/数据集处理/1906拍拍/"
outpath = "F:/项目相关/1906paipai/sub/"

test = pd.read_csv(open(oripath+"test.csv",encoding='utf8')) #
model_33 = pd.read_csv(open(outpath+"sub_lgb_33_0619_noprovince_mx4.csv",encoding='utf8'),parse_dates=["repay_date"]) #
model_33 = model_33.merge(test[["listing_id","due_amt"]],how='left',on='listing_id')

model_33['rank'] = model_33.groupby(['listing_id'])['repay_date'].rank(ascending=False,method='first')

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
    if 'rank' not in model.columns:
        model['repay_date']=pd.to_datetime(model['repay_date'])
        model['rank'] = model.groupby(['listing_id'])['repay_date'].rank(ascending=False, method='first')
    if 'due_amt' not in model.columns:
        model = model.merge(test[["listing_id", "due_amt"]], how='left', on='listing_id')
    if class_ratio=="last_date":
        c = 0
    elif class_ratio=="overdue":
        c = 32
    else:
        c = int(class_ratio.split("_")[-1])
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

auc = pd.read_csv(open(outpath+"auc_model1_31_0619.csv",encoding='utf8'))

# no032
# #对0覆盖
# is_last_date = pd.read_csv(open(outpath+"is_last_date0613.csv",encoding='utf8')) #
# model_33_no032 = cover(model_33_no032,is_last_date)
# #对32覆盖
# is_overdue = pd.read_csv(open(outpath+"is_overdue0613.csv",encoding='utf8')) #
# model_33_no032 = cover(model_33_no032,is_overdue)
#
# model_33_no032[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_lgbno032_cover032_0614.csv",index=None)
########################################
date = '0619'
#33模型 对0覆盖.auc*0.5为最优 6404
model_33 = pd.read_csv(open(outpath+"sub_lgb_10_0613.csv",encoding='utf8'))#8022
model_33 = pd.read_csv(open(outpath+"sub_lgb_33_0619_noprovince_mx4.csv",encoding='utf8'))#7022
model_33 = pd.read_csv(open(outpath+"sub_lgb_33_0613.csv",encoding='utf8')) #7211
n=0
is_last_date = pd.read_csv(open(outpath+"is_last_date{}.csv".format(date),encoding='utf8')) #lgb
is_last_date2 = pd.read_csv(open(outpath+"is_last_date_dfm.csv".format(date),encoding='utf8')) # dfm
is_last_date = is_last_date.merge(is_last_date2,how='left',on="listing_id")
is_last_date["rank1"] = is_last_date["last_date"].rank(method='first')
is_last_date["rank2"] = is_last_date["is_last_date"].rank(method='first')
# is_last_date["last_date"]=0.3*is_last_date["rank1"]+0.7*is_last_date["rank2"]
is_last_date["last_date"]=is_last_date["is_last_date"]
is_last_date = is_last_date[["user_id","listing_id","auditing_date","due_amt","last_date"]]
model_33 = cover(model_33,is_last_date,threshold=0.5*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_lgb_cover0_auc05_{}dfm.csv".format(date),index=None)
# model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_rank37_cover0_auc05_{}.csv".format(date),index=None)

#对32覆盖.auc*0.5为最优
n=32
is_overdue = pd.read_csv(open(outpath+"is_overdue{}.csv".format(date),encoding='utf8')) #lgb
is_overdue2 = pd.read_csv(open(outpath+"is_overdue_dfm.csv".format(date),encoding='utf8')) #dfm
is_overdue = is_overdue.merge(is_overdue2,how='left',on="listing_id")
is_overdue["rank1"] = is_overdue["overdue"].rank(method='first')
is_overdue["rank2"] = is_overdue["is_overdue"].rank(method='first')
is_overdue["overdue"]=(is_overdue["rank1"]+is_overdue["rank2"])/2
is_overdue = is_overdue[["user_id","listing_id","auditing_date","due_amt","overdue"]]
model_33 = cover(model_33,is_overdue,threshold=0.03*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_lgb_cover32_auc003_{}.csv".format(date),index=None)

#拼接model_33_no032和model_33，对最有可能为1~31的数据替换为model_33_no032
#1. 0.408187+0.1 last_date 和 0.117192+0.05 overdue id（获得最可能为0和32的ID）
#2. 以覆盖了auc阈值的model_33为基础，其余ID均认为是1~31的ID 拼入 model_33_no032
# is_mid = pd.read_csv(open(outpath+"is_mid0616.csv",encoding='utf8'))
# is_mid["rank"] = is_mid['is_mid'].rank(ascending=False,method='first')
# threshold_mid = test.shape[0]*(1-dic[0]-dic[32])*0.79*0.5#auc0.79
# print(threshold_mid)
# mid_id = set(is_mid[is_mid["rank"]<=threshold_mid]["listing_id"])
# model_33.loc[model_33["listing_id"].isin(mid_id)==True,'repay_amt']= \
#     model_33_no032.loc[model_33_no032["listing_id"].isin(mid_id)==True]['repay_amt']
# model_33[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover032_0616.csv",index=None)
#######################################################
date = "0619"
# todo 对1~31进行覆盖
#正常训练覆盖0 32 6404
ans1 = pd.read_csv(open(outpath+"sub_lgb_cover0_auc05_{}dfm.csv".format(date),encoding='utf8'))#6404
ans1 = pd.read_csv(open(outpath+"sub_lgb_10_0613.csv",encoding='utf8'))#8022
#1 0.5 6900 0.2 6523
n=1
is_1 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_1,threshold=0.05*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}005_0616.csv".format(n),index=None)

#2
n=2
is_2 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_2,threshold=0.05*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}005_0616.csv".format(n),index=None)

n=3#效果不佳
is_3 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_3,threshold=0.05*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}005_0616.csv".format(n),index=None)

n=4#
is_4 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_4,threshold=0.1*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}01_0616.csv".format(n),index=None)

n=5#
is_5 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_5,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=6#
is_6 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_6,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=7#
is_7 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_7,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=8#
is_8 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_8,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=9#
is_9 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_9,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=10#
is_10 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_10,threshold=0.3*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}03_0616.csv".format(n),index=None)

n=11#
is_11 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_11,threshold=0.3*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}03_0616.csv".format(n),index=None)

n=12#
is_12 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_12,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

n=13#未单测
is_13 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_13,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=14#
is_14 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_14,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=15#
is_15 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_15,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=16#
is_16 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_16,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
# ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_0616.csv".format(n),index=None)

# n=17#
# is_17 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_17,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=18#
# is_18 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_18,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=19#
# is_19 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_19,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=20#
# is_20 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_20,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=21#
# is_21 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_21,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=22#
# is_22 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_22,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=23#
# is_23 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_23,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=24#
# is_24 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_24,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
#
# n=25#
# is_25 = pd.read_csv(open(outpath+"is_{}_0614.csv".format(n),encoding='utf8')) #
# ans1 = cover(ans1,is_25,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=26#
is_26 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_26,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=27#
is_27 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_27,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=28#
is_28 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_28,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=29#
is_29 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_29,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=30#
is_30 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_30,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])

n=31#
is_31 = pd.read_csv(open(outpath+"is_{}_{}.csv".format(n,date),encoding='utf8')) #
ans1 = cover(ans1,is_31,threshold=0.2*dic[n]*auc.loc[auc.model==n,'auc'].values[0])
ans1[["listing_id","repay_amt","repay_date"]].to_csv(outpath+"sub_stack_cover{}02_{}dfm.csv".format(n,date),index=None)

