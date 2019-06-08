# -*- coding: utf-8 -*-
#@author: li
#@file: rule.py
#@time: 2019/6/7 16:58
"""
文件说明：规则构建
"""
import pandas as pd
path = "F:/数据集/1906拍拍/"
outpath = "F:/项目相关/1906拍拍/out/"
test = pd.read_csv(open(path+"test.csv",encoding='utf8')) #13W
submission = pd.read_csv(open(path+"submission.csv",encoding='utf8'))

#所有人均当天还款 线上28399.990100
# submission["repay_amt"]=0
# sub = submission.merge(test[["listing_id","due_date","due_amt"]], how='left', left_on=['listing_id',"repay_date"],right_on=["listing_id","due_date"])
# sub = sub.fillna(0)
# sub["repay_amt"] = sub["due_amt"]
# sub[["listing_id","repay_amt","repay_date"]].to_csv("F:/项目相关/1906拍拍/out/rule_all_due_amt0607.csv",index=None)

#所有人每天都还款,每天还款额为应还的加权,按整体加权比28/31分别加权效果略好，
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
31:0.008368}
dic28={
0:0.360304,
1:0.148709,
2:0.072087,
3:0.071244,
4:0.030712,
5:0.025466,
6:0.019713,
7:0.017314,
8:0.013941,
9:0.011562,
10:0.010737,
11:0.008732,
12:0.007964,
13:0.007083,
14:0.00609,
15:0.005453,
16:0.004928,
17:0.005134,
18:0.004347,
19:0.004366,
20:0.003504,
21:0.003598,
22:0.003373,
23:0.003242,
24:0.003073,
25:0.003017,
26:0.003748,
27:0.005865,
28:0.01396,
}
dic31={
0:0.408204,
1:0.118253,
2:0.058718,
3:0.056421,
4:0.026702,
5:0.02136,
6:0.017466,
7:0.014851,
8:0.013118,
9:0.01142,
10:0.010078,
11:0.009008,
12:0.008291,
13:0.007863,
14:0.006925,
15:0.006455,
16:0.006174,
17:0.005892,
18:0.005594,
19:0.005095,
20:0.004916,
21:0.004566,
22:0.004266,
23:0.004148,
24:0.003882,
25:0.003838,
26:0.00384,
27:0.003941,
28:0.003922,
29:0.004494,
30:0.007428,
31:0.01456,
}
submission["repay_amt"]=0
test["month"] = test["auditing_date"].apply(lambda x:int(x[5:7]))
sub = submission.merge(test[["listing_id","due_date","due_amt","month"]], how='left', left_on=['listing_id'],right_on=["listing_id"])
sub["repay_date_d"] = pd.to_datetime(sub["repay_date"])
sub["due_date_d"] = pd.to_datetime(sub["due_date"])
sub["day_diff"] = (sub["due_date_d"]-sub["repay_date_d"]).apply(lambda x:x.days)
sub["weight"] = sub["day_diff"].replace(dic)
# sub["weight"] = 0
# sub.loc[sub["month"]==2,"weight"] = sub.loc[sub["month"]==2,"day_diff"].replace(dic28)
# sub.loc[sub["month"]==3,"weight"] = sub.loc[sub["month"]==3,"day_diff"].replace(dic31)
sub["repay_amt"] = sub["due_amt"]*sub["weight"]
sub[["listing_id","repay_amt","repay_date"]].to_csv("F:/项目相关/1906拍拍/out/rule_weight_due_amt0607.csv",index=None)
