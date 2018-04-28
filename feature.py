#coding:utf-8
import pandas as pd
import numpy as np
import sys

df = pd.read_csv('./data/LoanStats3a.csv', skiprows = 1,low_memory=False)
df.drop('id', 1, inplace = True)
df.drop('member_id', 1, inplace = True)
df.term.replace(to_replace= '[^0-9]+', value = '', inplace = True, regex =True)#在方括号中表示不接受
df.int_rate.replace('%', '', inplace = True)#如果删不掉就说明是数值型，不必去删除
df.drop('sub_grade', 1, inplace = True)
df.drop('emp_title', 1, inplace = True)
df.emp_length.replace('n/a', np.nan, inplace = True)
df.emp_length.replace(to_replace= '[^0-9]+', value = '', inplace = True, regex =True)
#去掉全部为空的特征及实例
df.dropna(axis = 1 , how='all', inplace= True)
df.dropna(axis = 0 , how='all', inplace= True)
#查看目前表格状态，以决定下一步怎么走
df.drop(['debt_settlement_flag_date','settlement_status','settlement_date',\
         'settlement_amount','settlement_percentage','settlement_term'], 1, inplace = True)
#已经删除空值较多的序列
#现在根据列属性(float, object)删除数据值较单一的列
# for col in df.select_dtypes(include = ['float']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))
#删除列中，元素值差异性很小的列
df.drop(['delinq_2yrs','inq_last_6mths','mths_since_last_delinq',\
         'mths_since_last_record','open_acc','pub_rec','total_acc',
         'out_prncp','out_prncp_inv',\
         'collections_12_mths_ex_med','policy_code','acc_now_delinq','chargeoff_within_12_mths',\
         'delinq_amnt','pub_rec_bankruptcies','tax_liens'], 1, inplace = True)
#=================================================================================
#现在删除object类型中数值较为单一的列：
# for col in df.select_dtypes(include = ['object']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))
      
df.drop(['term','grade','emp_length',\
         'home_ownership','verification_status','issue_d','pymnt_plan',
         'purpose','zip_code',\
         'addr_state','earliest_cr_line','initial_list_status','last_pymnt_d',\
         'next_pymnt_d','last_credit_pull_d','application_type',\
         'hardship_flag','disbursement_method','debt_settlement_flag'], 1, inplace = True)
#现在再次打印信息，以确定下一步怎么办?
df.drop('desc', 1, inplace = True)
#对Y值进行处理
df.loan_status.replace('Fully Paid', int(1), inplace = True)
df.loan_status.replace('Charged Off', int(0), inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid', np.nan, inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off', np.nan, inplace = True)
df.dropna(subset = ['loan_status'], inplace = True)
df.drop('title', 1, inplace = True)
df.fillna(0,inplace = True)
df.fillna(0.0,inplace = True)
#删除相关系数0.95以上的值
df.drop(['loan_amnt','funded_amnt','total_pymnt'] ,1, inplace = True)
#18个特征值进行正交处理
cor = df.corr()
cor.iloc[:,:] = np.tril(cor, k = -1)
cor = cor.stack()
print(cor[(cor>0.55) | (cor<-0.55)])
df = pd.get_dummies(df)
df.to_csv('./data/feature01.csv')
print(df.loan_status.value_counts())
 
# print(df.info())




