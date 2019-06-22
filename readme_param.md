### lgb 
0613
全特征 学习率0.02 7211
0619
全特征 学习率0.04 2.05467 7318
全特征 学习率0.02 2.05394 7334
去除pred 2.03105 7024
去除pred province 2.02949 7017
去除pred province 增加全部用户tag 2.03093 7029
### xgb
0613
全特征 2.04209 7200

0619


### 融合
lgb xgb 55 7202
lgb0619 xgb 7229

### 从二分类检查有问题的特征
is_overdue
0613 0.7679
0613+train 0.7628
0613+behavior 07628
0613+listing 0.7629
0613+repay_logs 0.7635
0613+user_info 0.7627
0613+other 0.68

0619 0.68
去除pred 0.7642
去除pred province 0.7641

is_last_date
0613 0.74

0619 
去除pred 0.7484