import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
import seaborn as sns

account_df = pd.read_csv('./raw/account.csv', delimiter=";")
disp_df = pd.read_csv('./raw/disp.csv', delimiter=';')
client_df = pd.read_csv('./raw/client.csv', delimiter=';')
district_df = pd.read_csv('./raw/district.csv', delimiter=';')
trans_df = pd.read_csv('./raw/trans_dev.csv', delimiter=';', low_memory=False)
loan_df = pd.read_csv('./raw/loan_dev.csv', delimiter=';')
card_df = pd.read_csv('./raw/card_dev.csv', delimiter=';')


def missing_data_imputation(df, column_name, index, cat=True):
    testdf = df[df[column_name].isnull()==True]
    traindf = df[df[column_name].isnull()==False]
    
    y = traindf[column_name]
    traindf_drop = traindf.drop(column_name,axis=1)
    
    model = LogisticRegression() if cat else LinearRegression()
    model.fit(traindf_drop,y)

    testdf = testdf.drop(column_name,axis=1)
    pred = model.predict(testdf)   
    testdf[column_name] = pred

    df = pd.concat([testdf, traindf], axis=0)

    return df

# Loans
# Converts date to datetime
loan_df['date'] = pd.to_datetime(loan_df['date'], format='%y%m%d')


# Account
# Converts date to datetime
account_df['date'] = pd.to_datetime(account_df['date'], format='%y%m%d')

# Encodes 'frequency' and transforms it to numerical values
encoder = preprocessing.LabelEncoder()
encoder.fit(account_df['frequency'].unique())
account_df['frequency'] = encoder.transform(account_df['frequency'])


# Disponent
disp_df.loc[disp_df["type"] == "OWNER",    "type"] = "O"
disp_df.loc[disp_df["type"] == "DISPONENT","type"] = "U"

# Transform numerical into categorical
disp_df = pd.get_dummies(disp_df)


# Card
# Converts date to datetime
card_df['date'] = pd.to_datetime(card_df['issued'], format='%y%m%d')
# Encodes 'type' and transforms it to numerical values
encoder = preprocessing.LabelEncoder()
encoder.fit(card_df['type'].unique())
card_df['type'] = encoder.transform(card_df['type'])


#Client
# Converts 'birth_number' to a readable date and drops redundant 'birth_number' column
client_df['gender'] = client_df.apply(lambda row: 1 if (row['birth_number']//100)%100 < 50 else 0, axis=1)
client_df['birth_date'] = client_df.apply(lambda row: "19{:02d}-{:02d}-{:02d}".format((row['birth_number']//10000)%100,((row['birth_number']//100)%100 if row['gender'] == 1 else (row['birth_number']//100)%100 - 50), (row['birth_number'])%100), axis=1)
client_df['birth_date'] = client_df['birth_date'].apply(pd.to_datetime)
client_df.drop('birth_number', inplace=True, axis=1)


#Transactions
# Converts date to datetime and then to ordinal number
trans_df['date'] = pd.to_datetime(trans_df['date'], format='%y%m%d')
trans_df['date'] = trans_df['date'].apply(pd.Timestamp.toordinal)

#If type is withdrawl than amount is negative
trans_df.loc[(trans_df["type"] == "withdrawal") | (trans_df["type"] == "withdrawal in cash"), "amount"] *= -1

# Encodes 'type' collumn to numerical values
encoder = preprocessing.LabelEncoder()
encoder.fit(trans_df['type'].unique())
trans_df['type'] = encoder.transform(trans_df['type'])

# Removes null values with mean under 70% by collumn
trans_df = trans_df[trans_df.columns[ trans_df.isnull().mean() < 0.7 ]]

## Null values imputation using logistic regression

missing_values_columns = ['operation', 'k_symbol']
for column in missing_values_columns:
    non_null_df = trans_df.copy()
    non_null_df = non_null_df.drop(columns=missing_values_columns)
    non_null_df[column] = trans_df[column] 

    non_null_df = missing_data_imputation(non_null_df, column, 'trans_id')
    trans_df[column] = non_null_df[column]

balance_stats = trans_df.sort_values(by=['account_id'], ascending=[True]).groupby(['account_id']).agg({'balance': ['mean', 'count', 'std', 'min', 'max']}).reset_index()
balance_stats.columns = ['account_id', 'balance_mean', 'balance_count', 'balance_std', 'balance_min', 'balance_max']

balance_stats.loc[balance_stats["balance_min"] >= 0, "negative_balance"] = 0
balance_stats.loc[balance_stats["balance_min"] < 0, "negative_balance"] = 1

amount_stats_type = trans_df.sort_values(by=['account_id'],ascending=[True]).groupby(['account_id', 'type']).agg({'amount': ['mean', 'count', 'std', 'max', 'min'],}).reset_index()

amount_stats_type_0 = amount_stats_type[amount_stats_type['type'] == 0]
amount_stats_type_1 = amount_stats_type[amount_stats_type['type'] == 1]
amount_stats_type_2 = amount_stats_type[amount_stats_type['type'] == 2]

amount_stats_type_0.columns = ['account_id', 'type', 'type_0_mean', 'type_0_count','type_0_max', 'type_0_min', 'type_0_std']
amount_stats_type_1.columns = ['account_id', 'type', 'type_1_mean', 'type_1_count','type_1_max', 'type_1_min', 'type_1_std']
amount_stats_type_2.columns = ['account_id', 'type', 'type_2_mean', 'type_2_count','type_2_max', 'type_2_min', 'type_2_std']

amount_stats_type_0.drop(['type'], axis=1, inplace=True)
amount_stats_type_1.drop(['type'], axis=1, inplace=True)
amount_stats_type_2.drop(['type'], axis=1, inplace=True)

type_df = amount_stats_type_0.merge(amount_stats_type_1, on='account_id', how='left').merge(amount_stats_type_2, on='account_id', how='left')

amount_stats_k_sym = trans_df.sort_values(by=['account_id'],ascending=[True]).groupby(['account_id', 'k_symbol']).agg({'amount': ['mean', 'count', 'std'],}).reset_index()

# k_symbol values: saction payment, household, pension, payment for statement, insurance payment, missing
sanctions   = amount_stats_k_sym[amount_stats_k_sym['k_symbol'] == 'sanction interest if negative balance']
households  = amount_stats_k_sym[amount_stats_k_sym['k_symbol'] == 'household']
pensions    = amount_stats_k_sym[amount_stats_k_sym['k_symbol'] == 'old-age pension']
payment_st  = amount_stats_k_sym[amount_stats_k_sym['k_symbol'] == 'payment for statement']
ins_payment = amount_stats_k_sym[amount_stats_k_sym['k_symbol'] == 'insurrance payment']

sanctions.columns   = ['account_id', 'k_symbol', 'sanctions_mean', 'sanctions_count', 'sanctions_std']
households.columns  = ['account_id', 'k_symbol', 'household_mean', 'household_count', 'household_std']
pensions.columns    = ['account_id', 'k_symbol', 'pension_mean', 'pension_count', 'pension_std']
payment_st.columns  = ['account_id', 'k_symbol', 'payment_statement_mean', 'payment_statement_count', 'payment_statement_std']
ins_payment.columns = ['account_id', 'k_symbol', 'ins_payment_mean', 'ins_payment_count', 'ins_payment_std']

sanctions.drop(['k_symbol'], axis=1, inplace=True)
households.drop(['k_symbol'], axis=1, inplace=True)
pensions.drop(['k_symbol'], axis=1, inplace=True)
payment_st.drop(['k_symbol'], axis=1, inplace=True)
ins_payment.drop(['k_symbol'], axis=1, inplace=True)

k_symbol_df = sanctions.merge(households, on='account_id', how='left').merge(pensions, on='account_id', how='left').merge(payment_st, on='account_id', how='left').merge(ins_payment, on='account_id', how='left')

operations = trans_df.sort_values(by='account_id', ascending=[True]).groupby(['account_id', 'operation']).agg({'amount': ['count', 'mean', 'max', 'std', 'min'],}).reset_index()
operations.columns = ['account_id', 'operation', 'amount_count', 'amount_mean','amount_max', 'amount_std', 'amount_min']
    
# Operations: credit_cash, col_another_bank, interest, withd_cash, rem_another_bank, credit_card_withd
credit_cash       = operations[operations['operation'] == 'credit in cash']
col_another_bank  = operations[operations['operation'] == 'collection from another bank']
interest          = operations[operations['operation'] == 'interest credited']
withd_cash        = operations[operations['operation'] == 'withdrawal in cash']
rem_another_bank  = operations[operations['operation'] == 'remittance to another bank']
credit_card_withd = operations[operations['operation'] == 'credit card withdrawal']

credit_cash.columns       = ['account_id', 'operation', 'credit_cash', 'credit_cash_mean', 'credit_cash_max', 'credit_cash_std', 'credit_cash_min']
col_another_bank.columns  = ['account_id', 'operation', 'col_another_bank', 'col_another_bank_mean', 'col_another_bank_max', 'col_another_bank_std', 'col_another_bank_min']
interest.columns          = ['account_id', 'operation', 'interest', 'interest_mean', 'interest_max', 'interest_std', 'interest_min']
withd_cash.columns        = ['account_id', 'operation', 'withd_cash', 'withd_cash_mean', 'withd_cash_max', 'withd_cash_std', 'withd_cash_min']
rem_another_bank.columns  = ['account_id', 'operation', 'rem_another_bank', 'rem_another_bank_mean', 'rem_another_bank_max', 'rem_another_bank_std', 'rem_another_bank_min']
credit_card_withd.columns = ['account_id', 'operation', 'credit_card_withd', 'credit_card_withd_mean', 'credit_card_withd_max', 'credit_card_withd_std', 'credit_card_withd_min']

credit_cash.drop(['operation'], axis=1, inplace=True)
col_another_bank.drop(['operation'], axis=1, inplace=True)
interest.drop(['operation'], axis=1, inplace=True)
withd_cash.drop(['operation'], axis=1, inplace=True)
rem_another_bank.drop(['operation'], axis=1, inplace=True)
credit_card_withd.drop(['operation'], axis=1, inplace=True)

operations_df = credit_cash.merge(col_another_bank, on='account_id', how='left').merge(interest, on='account_id', how='left').merge(withd_cash, on='account_id', how='left').merge(rem_another_bank, on='account_id', how='left').merge(credit_card_withd, on='account_id', how='left')

df = balance_stats.merge(type_df, on='account_id', how='left').merge(k_symbol_df, on='account_id', how='left').merge(operations_df, on='account_id', how='left')
df.fillna(value=0, inplace=True)

# Generating some features
df['total operations']                = df['credit_cash'] + df['col_another_bank'] + df['withd_cash'] + df['rem_another_bank'] + df['credit_card_withd'] + df['interest']
df['mean_trans_amount']        = df['type_0_mean']      + df['type_1_mean'] + df['type_2_mean']
df['delta_balance']            = df['balance_max']      - df['balance_min']
df['ratio_credit']             = df['type_0_count']     / df['total operations']
df['ratio_withdrawl']         = (df['type_1_count'] +  df['type_2_count'])  / df['total operations']
df['ratio_credit_cash']        = df['credit_cash']      / df['total operations']
df['ratio_col_another_bank']   = df['col_another_bank'] / df['total operations']
df['ratio_withd_cash']         = df['withd_cash']       / df['total operations']
df['ratio_rem_another_bank']   = df['rem_another_bank'] / df['total operations']
df['ratio_credit_card_withd']  = df['credit_card_withd']/ df['total operations']
df['ratio_interest']           = df['interest']         / df['total operations']

trans_df = df

# District
# Encodes 'region' collumn to numerical values
encoder = preprocessing.LabelEncoder()
encoder.fit(district_df['region'].unique())
district_df['region'] = encoder.transform(district_df['region'])

# Drops repeated column 'name'
district_df = district_df.drop('name ', axis=1)

district_df.loc[district_df['no. of commited crimes \'95 ']=="?", 'no. of commited crimes \'95 '] = None
district_df.loc[district_df['unemploymant rate \'95 ']=="?", 'unemploymant rate \'95 '] = None

missing_values_columns = ['no. of commited crimes \'95 ', 'unemploymant rate \'95 ']
for column in missing_values_columns:
    non_null_df = district_df.copy()
    non_null_df = non_null_df.drop(columns=missing_values_columns)
    non_null_df[column] = district_df[column] 

    non_null_df = missing_data_imputation(non_null_df, column, 'code', cat=False)
    district_df[column] = non_null_df[column]

district_df['unemploymant rate \'95 ']      = district_df['unemploymant rate \'95 '].astype(float)
district_df['no. of commited crimes \'95 '] = pd.to_numeric(district_df['no. of commited crimes \'95 '])
district_df['unemploymant rate \'96 ']      = district_df['unemploymant rate \'96 '].astype(float)
district_df['no. of commited crimes \'96 '] = pd.to_numeric(district_df['no. of commited crimes \'96 '])
district_df['no. of enterpreneurs per 1000 inhabitants '] = pd.to_numeric(district_df['no. of enterpreneurs per 1000 inhabitants '])
district_df['ratio of urban inhabitants ']  = district_df['ratio of urban inhabitants '].astype(float)
district_df['ratio entrepeneurs']           = district_df['no. of enterpreneurs per 1000 inhabitants '] / 1000
district_df['ratio of urban inhabitants ']  = district_df['ratio of urban inhabitants '] / 100
district_df['crime_delta']                  = (district_df['no. of commited crimes \'96 '] - district_df['no. of commited crimes \'95 ']) / district_df['no. of inhabitants']
district_df['unemploymant_delta']           = district_df['unemploymant rate \'96 '] - district_df['unemploymant rate \'95 ']
    
district_df = district_df.drop('no. of enterpreneurs per 1000 inhabitants ', axis=1)
district_df = district_df.rename(columns={'code ': 'district_id'})


# Merge
main_df = loan_df.merge(account_df.rename(columns={'date': 'acc_create_date'}), on='account_id', how='left').merge(trans_df, on='account_id', how='left')
main_df = main_df.drop(['district_id'], axis=1) 


df_disp_client_card = disp_df.merge(client_df, on='client_id', how='left').merge(card_df, on='disp_id', how='left')
df_disp_client = df_disp_client_card.drop(['issued'], axis=1)
df_disp_client .fillna(value=-1, inplace=True)

def get_first(df): return df.iloc[0]
df_disp_client = df_disp_client.sort_values(by=['account_id', 'type_O'], ascending=[True, False]).groupby(['account_id']).agg({'type_O': ['count'],'type_U': ['count'],'gender': get_first,'birth_date': get_first,'district_id': get_first,'type': get_first}).reset_index()
df_disp_client.columns= ['account_id', 'owner_count', 'disponent_count', 'owner_gender', 'owner_birthdate', 'district_id', 'card_type']
df_disp_client['has_card'] = df_disp_client['card_type']
df_disp_client.loc[df_disp_client["card_type"] >= 0, "has_card"] = 1
df_disp_client.loc[df_disp_client["card_type"] <  0, "has_card"] = 0

df = main_df.merge(df_disp_client.merge(district_df, on='district_id'), on='account_id')

df['acc_age_on_loan'] = (df['date'] - df['acc_create_date']).dt.days    
df['acc_age_mths'] = df['acc_age_on_loan'] / 30
df["trans_mth"] = df['total operations'] / df["acc_age_mths"]
df['withdrawal_mth'] = (df['type_1_mean'] + df['type_2_mean'])  / df["acc_age_mths"]
df['credit_mth']  = df['type_0_mean'] / df["acc_age_mths"]
df['avg_mth_income'] = (df['average salary '] + df['household_mean']) / 12
df['mean_income_disp'] = df['type_0_mean'] + df['type_1_mean'] + df['type_2_mean']
df['real_mth_income'] =  df['mean_income_disp'] + (df['household_mean'] / 12)
df.loc[df["pension_mean"] > 0, "avg_mth_income"] = (df["pension_mean"] + df['household_mean'])               / 12
df.loc[df["pension_mean"] > 0, "real_mth_income"] = df["real_mth_income"] + (df['pension_mean']                 / 12)
df['ratio_real_salary_to_expected'] = df['real_mth_income']/ df['avg_mth_income']
df['ratio_withd_credit_mth'] = df['withdrawal_mth'] / df['credit_mth']
df['owner_age_on_loan'] = (df['date'] - df['owner_birthdate']).dt.days / 365
df.loc[df["owner_age_on_loan"] < 18, "underage"] = 1
df.loc[df["owner_age_on_loan"] >=  18, "underage"] = 0
df["ratio_max_value_in_account_to_loan"] = df["balance_max"] / df["amount"]
df['ratio_expected_income_to_payments'] = df['avg_mth_income'] / df['payments']
df['real_income_to_payments_ratio'] = df['real_mth_income'] / df['payments']

df.drop(['account_id', 'district_id'], inplace=True, axis=1)
print(df.shape)
# Create correlation matrix
corr_matrix = df.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]



# Drop features - Filter based
df.drop(to_drop, axis=1, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].apply(pd.Timestamp.toordinal)

df['acc_create_date'] = pd.to_datetime(df['acc_create_date'])
df['acc_create_date'] = df['acc_create_date'].apply(pd.Timestamp.toordinal)

df['owner_birthdate'] = pd.to_datetime(df['owner_birthdate'])
df['owner_birthdate'] = df['owner_birthdate'].apply(pd.Timestamp.toordinal)


# Drop features - Wrapper based
input_names = list(df.columns)
input_names.remove('status')
input_names.remove('loan_id')

y = df['status']
X = df[input_names]

sfs = SFS(LogisticRegression(max_iter=1000),
           k_features='best',
           forward=True,
           floating=False,
           scoring = 'roc_auc',
           cv = 0,
           n_jobs=-1)        

sfs.fit(X,y)

selection = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
selected = list(selection.sort_values(by=["cv_scores"], ascending=False)["feature_names"].iloc[0])

df = df[["loan_id"] + selected + ["status"]]


csv = df.to_csv(index = False)
os.makedirs("processed", exist_ok=True)
path = "{}/{}".format("processed", "data_6it.csv")
with open(path, 'w') as fd: fd.write(csv)