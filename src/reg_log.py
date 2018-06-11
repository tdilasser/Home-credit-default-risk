#!/home/el-diablo/anaconda3/bin/python3

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import roc_auc_score

data_application_test = pd.read_csv('../data/application_test.csv')
data_application_train = pd.read_csv('../data/application_train.csv')
# data_bureau_balance = pd.read_csv('../data/bureau_balance.csv')
# data_credit_card_balance  = pd.read_csv('../data/credit_card_balance.csv')
# data_bureau = pd.read_csv('../data/bureau.csv')
# data_HomeCredit_columns_description = pd.read_csv('../data/HomeCredit_columns_description.csv')
# data_installments_payments = pd.read_csv('../data/installments_payments.csv')
# data_POS_CASH_balance = pd.read_csv('../data/POS_CASH_balance.csv')
# data_previous_application = pd.read_csv('../data/previous_application.csv')
data_sample_submission = pd.read_csv('../data/sample_submission.csv')


data_application_train.columns = [str.lower(x) for x in data_application_train.columns]
data_application_test.columns = [str.lower(x) for x in data_application_test.columns]

#data_application_train.head()

app_train = pd.get_dummies(data_application_train)

app_train = app_train.dropna()


target = app_train.target
app_train.drop(['sk_id_curr','target'],axis=1,inplace=True)


X_train, X_test, y_train, y_test = train_test_split(app_train, target, test_size=0.4, random_state=1)

regleg = LogisticRegression()
pred = regleg.fit(X_train,y_train).predict_proba(X_test)

plot = (skplt.metrics.plot_roc(y_test, pred))
plt.savefig('roc.png')
print(roc_auc_score(y_test, pred[:,1]))


#test = pd.get_dummies(data_application_test)
#app_train.columns.difference(test.columns)
#
#a = np.setdiff1d(app_train.columns, test.columns)
#
#test['code_gender_XNA'] = 0
#test['name_family_status_Unknown'] = 0
#test['name_income_type_Maternity leave'] = 0
#
#test.fillna(0,inplace=True)
#test.drop('sk_id_curr',axis=1,inplace=True)
#
#prediction = regleg.predict_proba(test)
#
#data_sample_submission['TARGET'] = prediction[:,1]
#
#data_sample_submission.to_csv('../baseline.csv',index=False)



