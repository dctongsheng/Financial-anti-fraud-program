#coding:utf-8


import pandas as pd
import numpy as np
import sys

df = pd.read_csv('./data/LoanStats3a.csv', skiprows = 1, low_memory = True)#skiprows跳过第一行，low_memory低内存加载，报错就该成False
'''读入接待信息'''
# print(df.head(10))
# print(df.info())
'''查看数据特征表格信息'''
df.drop('id', axis = 1, inplace = True)
df.drop('member_id', axis = 1, inplace = True)

'''删除了肉眼看的见的空值列'''
df.term.replace(to_replace = '[^0-9]+', value = '', inplace = True, regex = True)#regex正则打开
df.int_rate.replace('%', value = '', inplace = True)#去不掉说明就是浮点型

'''清洗数据，去除特征中的特殊字符'''
df.drop('sub_grade', axis = 1, inplace = True)
df.drop('emp_title', axis = 1, inplace = True)

df.emp_length.replace('n/a', np.nan, inplace = True)
df.emp_length.replace(to_replace = '[^0-9]+', value = '', inplace = True, regex = True)
#这一步是必须做的，这样做以后才能，用info查看
df.dropna(axis = 1, how = 'all', inplace = True)
df.dropna(axis = 0, how = 'all', inplace = True)

'''删除空值较多的列'''
'''debt_settlement_flag_date     98 non-null object
settlement_status             155 non-null object
settlement_date               155 non-null object
settlement_amount             155 non-null float64
settlement_percentage         155 non-null float64
settlement_term               155 non-null float64'''
df.drop(['debt_settlement_flag_date','settlement_status','settlement_date',\
         'settlement_amount','settlement_percentage',\
         'settlement_term'], axis = 1, inplace = True)
'''删除不为空，但是重复较多的列；先删float，再删object'''

# for col in df.select_dtypes(include = ['float']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))

'''
col delinq_2yrs has 13
col inq_last_6mths has 29
col mths_since_last_delinq has 96
col mths_since_last_record has 114
col open_acc has 45
col pub_rec has 7

col total_acc has 84
col out_prncp has 1
col out_prncp_inv has 1

col collections_12_mths_ex_med has 2
col policy_code has 1
col acc_now_delinq has 3
col chargeoff_within_12_mths has 2
col delinq_amnt has 4
col pub_rec_bankruptcies has 4
col tax_liens has 3
'''
df.drop(['delinq_2yrs','inq_last_6mths','mths_since_last_delinq',\
         'mths_since_last_record','open_acc','pub_rec','total_acc',\
         'out_prncp','out_prncp_inv','collections_12_mths_ex_med',\
         'policy_code','acc_now_delinq','chargeoff_within_12_mths',\
         'delinq_amnt','pub_rec_bankruptcies',\
         'tax_liens'], axis = 1, inplace = True)

'''删除objetct类型中数据重复较多的值'''
# for col in df.select_dtypes(include = ['object']).columns:
    # print('col {} has {}'.format(col, len(df[col].unique())))

'''
col term has 2
col grade has 7
col emp_length has 11
col home_ownership has 5
col verification_status has 3
col issue_d has 55

col pymnt_plan has 1
col purpose has 1
col zip_code has 837
col addr_state has 50
col earliest_cr_line has 531
col initial_list_status has 1

col last_pymnt_d has 113
col next_pymnt_d has 99
col last_credit_pull_d has 125
col application_type has 1
col hardship_flag has 1
col disbursement_method has 1
col debt_settlement_flag has 2
''' 
df.drop(['term','grade','emp_length','home_ownership','verification_status'\
         ,'issue_d','pymnt_plan','purpose','zip_code','addr_state',\
         'earliest_cr_line','initial_list_status','last_pymnt_d',\
         'next_pymnt_d','last_credit_pull_d','application_type','hardship_flag',
         'disbursement_method','debt_settlement_flag'], axis = 1, inplace = True)

df.drop(['desc','title'], axis = 1, inplace = True)

# df.to_csv('./df_data.csv')
# print(df.loan_status.value_counts())
'''标签二值化'''

df.loan_status.replace('Fully Paid', value = int(1), inplace = True)
df.loan_status.replace('Charged Off', value = int(0), inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid', \
                       np.nan, inplace = True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off', \
                       np.nan, inplace = True)
'''删除标签为空的实力，大概删除了3000个不到的实力'''
df.dropna(subset = ['loan_status'], how = 'any', inplace = True)
# print(df.loan_status.value_counts())
# print(df.info())
# sys.exit(0)
# print(df.head(10))
'''把样本中的空值用0.0去填充'''
df.fillna(0.0, inplace = True)

'''==========================以上部分为数据清洗==========================='''
'''计算清洁后样本数据的相关性'''
# cor = df.corr()#协方差矩阵
# cor.iloc[:, :] = np.tril(cor, k= -1)
# cor = cor.stack()
# print(cor[(cor>0.55)|(cor<-0.55)])
# sys.exit(0)
'''loan_amnt
funded_amnt
total_pymnt'''
'''删除相关系数大于0.95的列'''
df.drop(['loan_amnt','funded_amnt','total_pymnt'], axis = 1, inplace = True)
print(df.info())#revol_util                 39786 non-null object"%"会默认为object,其实他是数值
# sys.exit(0)
df = pd.get_dummies(df)#哑变量
df.to_csv('./data/feature03.csv')

# print(df.info())





