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
#对0覆盖
threshold = 0.5
temp = is_last_date.loc[is_last_date["last_date"]>threshold]
temp = temp["listing_id","due_amt"]
temp.columns = ["listing_id","due_amt_last_date"]
model_33 = model_33.merge(temp,how='left',on=["listing_id"])
model_33.loc[model_33["due_amt_last_date"]>=0,'repay_amt']=0
model_33.loc[(model_33["due_amt_last_date"]>=0)&(model_33["rank"]==1),'repay_amt']=model_33.loc[(model_33["due_amt_last_date"]>=0)&(model_33["rank"]==1)]["due_amt_last_date"]
#对32覆盖
