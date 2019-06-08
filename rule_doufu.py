import pandas as pd
import numpy as np
import datetime
'''
基本思想是假设用户一次性还钱，这样我就想知道用户历史
会在截止日期前多久还钱？其实，可能有些日期就是用户发工资的日子吧。
哈哈哈，猜测，猜测。
可以用log提取历史+train作为label进行建模吧。也许可以。
公众号：麻婆豆腐AI
'''
path = '../data/'
# user_id,listing_id,auditing_date,due_date,due_amt,repay_date,repay_amt
train = pd.read_csv(path + 'user_repay_logs.csv')
# user_id,listing_id,auditing_date,due_date,due_amt
test = pd.read_csv(path + 'test.csv')
#
train = train.sort_values(['listing_id','repay_date'])

train['due_date'] = pd.to_datetime(train['due_date'])
train['repay_date'] = train['repay_date'].replace("\\N",'2020-01-01')
train['repay_date'] = pd.to_datetime(train['repay_date'])

train['time_long'] = train['repay_date'] - train['due_date']
train['time_long'] = train['time_long'].dt.days
train['repay_amt'] = train['repay_amt'].replace("\\N",0)
train['money'] = train['repay_amt'] - train['due_amt']

test['auditing_date'] = pd.to_datetime(test['auditing_date'])
test['due_date'] = pd.to_datetime(test['due_date'])

test['repay_date'] = ''
test['repay_amt'] = ''

train_user_id = train['user_id'].unique()
test_copy = test[test['user_id'].isin(train_user_id)]

train_copy = train[train['user_id'].isin(test['user_id'].unique())]

# 这里可以随便修改，我是假设了一个情况，如果为还钱，给了10的权重。
# 这个部分的思路就是，如果你有过一次懈怠，那么可能下一次还会懈怠吧。哈哈，惰性思维。
# 你们可以构造一个线下，看看这个权重最好是什么时候，其实就是防止时间穿越，把提取的信息放到train里面。
train_copy['time_long'] = train_copy['time_long'].apply(lambda x:x if x<=0 else 10)

train_copy_1 = train_copy[['user_id','time_long']]
train_copy_1 = train_copy_1.groupby('user_id')['time_long'].mean().reset_index()#所有账单还款提前时间
train_copy_1.columns = ['user_id','time_s']
train_copy_1['time_s'] = train_copy_1['time_s'].apply(lambda x:np.int(x))
test_1 = pd.merge(test,train_copy_1,on=['user_id'],how='left')
test_1['time_s'] = test_1['time_s'].fillna(0)
test_1['time_s'] = test_1['time_s'].apply(lambda x:datetime.timedelta(days=x))
test_1['repay_date'] = test_1['due_date'] + test_1['time_s']
test_1['repay_amt1'] = test_1['repay_date'] > test_1['due_date']
test_1['repay_amt2'] = test_1['repay_date'] < test_1['auditing_date']

test_1['repay_amt_1'] = test_1['due_amt']

a = []
b = []
c = []

for i in test_1.values:
    if i[8] == True:
        a.append(i[1])
        b.append('')
        c.append('')
    elif i[9] == True:
        a.append(i[1])
        b.append(i[-1])
        c.append(i[3])
    else:
        a.append(i[1])
        b.append(i[-1])
        c.append(i[5])
# listing_id,repay_amt,repay_date
res = pd.DataFrame()
res['listing_id'] = a
res['repay_amt'] = b
res['repay_date'] = c
res[['listing_id','repay_amt','repay_date']].to_csv('../submit/submission.csv',index=False)