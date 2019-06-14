# -*- coding: utf-8 -*-
#@author: li
#@file: rule.py
#@time: 2019/6/7 16:58
"""
文件说明：规则构建
"""
import pandas as pd
path = "F:/数据集/1906拍拍/"
outpath = "F:/项目相关/1906paipai/sub/"

model_33 = pd.read_csv(open(outpath+"sub_lgb_33_0613.csv",encoding='utf8')) #
model_33_no032 = pd.read_csv(open(outpath+"lgb_33_no032_0613.csv",encoding='utf8')) #
is_last_date = pd.read_csv(open(outpath+"is_last_date0613.csv",encoding='utf8')) #
is_overdue = pd.read_csv(open(outpath+"is_overdue0613.csv",encoding='utf8')) #

model_33['rank'] = model_33.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
model_33_no032['rank'] = model_33_no032.groupby('listing_id')['repay_date'].rank(ascending=False,method='first')
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
#对0覆盖
threshold = 0.5
temp = is_last_date.loc[is_last_date["last_date"]>threshold]
temp = temp["listing_id","due_amt"]
temp.columns = ["listing_id","due_amt_last_date"]
model_33 = model_33.merge(temp,how='left',on=["listing_id"])
model_33.loc[model_33["due_amt_last_date"]>=0,'repay_amt']=0
model_33.loc[(model_33["due_amt_last_date"]>=0)&(model_33["rank"]==1),'repay_amt']=model_33.loc[(model_33["due_amt_last_date"]>=0)&(model_33["rank"]==1)]["due_amt_last_date"]
#对32覆盖
