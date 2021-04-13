import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import date_range
import seaborn as sns
from seaborn.regression import lmplot
from sklearn import metrics # calculate MSE and RMSE for linear regression
import swifter
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import curve_fit
from itertools import compress

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor, plot_tree

# automatically select the number of features for RFE


from sklearn.metrics import classification_report, confusion_matrix




# df limited to athletes who have pr offset less than 1.5 minutes
#  92271 men, 72779 women, 165,050 total
df_m = pd.read_pickle('men_df.pkl')
df_w = pd.read_pickle('women_df.pkl')
list_kj = ['year', 'meet', 'distance', 'location', 'name', 'decimaltime', 'pr', 'previous number races', 'szn pr', 'last race', 'avg previous races']
list2_kj = ['year', 'meet', 'distance', 'location', 'name', 'decimaltime', 'decimaltime hayward', 'days since pr', 'szn pr handicap', 'days since szn pr', 'last race handicap' ]


df_w_slow = df_w[df_w['szn_pr <= 19.308'] == 0]
df_w_fast = df_w[df_w['szn_pr <= 19.308'] == 1]
df_m_slow = df_m[df_m['szn_pr <= 16.275'] == 0]
df_m_fast = df_m[df_m['szn_pr <= 16.275'] == 1]



'''
The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building 
a model on those attributes that remain.It uses the model accuracy to identify which attributes 
(and combination of attributes) contribute the most to predicting the target attribute.
'''
#11


drop_list = ['year', 'start', 'meet', 'distance', 'gender', 'location', 'athlete_id',
       'place', 'name', 'class', 'school', 'originaltime', 'race_uid', 'state', 'region', 
       'sig location', 'sig locations top5avg', 'push', 'push ratio', 'second place push', 
       'second place push ratio', 'decimaltime hayward', 'off szn pr if not last szn pr']


# overall men
df_modeling = df_m.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
dtree = DecisionTreeRegressor()
rfe = RFE(lm,10)
fit = rfe.fit(X_train,y_train) 
lm_predictions = fit.predict(X_test)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

res_men = list(compress(X.columns, fit.support_))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))
# ****************************************************************************
#  MEN FAST
# ****************************************************************************

# men fast
df_modeling = df_m_fast.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
dtree = DecisionTreeRegressor()
rfe = RFE(lm,1)
fit = rfe.fit(X_train,y_train) 
lm_predictions = fit.predict(X_test)

res_men_fast = list(compress(X.columns, fit.support_))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))


df_modeling = df_m_fast.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
m_fast_list = ['decimaltime', 'avg previous races', 'szn avg', 'pr']
df_modeling = df_modeling[m_fast_list]
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


X_trainC = sm.add_constant(X_train)
lr=  sm.OLS(y_train, X_trainC)
olsres = lr.fit()
X_testC = sm.add_constant(X_test)
ols_predictions = olsres.predict(X_testC)
print(olsres.summary())

lm = LinearRegression()
lm.fit(X_train,y_train) 
lm_predictions = lm.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, lm_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

lm.intercept_
coefficents = lm.coef_
coeff_df = pd.DataFrame(coefficents,X.columns,columns=['Coefficient'])

df_predicted = X_test
df_predicted['decimal time'] = y_test
df_predicted['predicted'] = lm_predictions
# residual is actual minus predicted. Positive residual means we predictions are low
df_predicted['residual'] =  df_predicted['decimal time'] - df_predicted['predicted']
df_temp = df_m_fast[['year', 'start','name', 'gender','school', 'hayward handicap', 'state']]
df_predicted = df_predicted.join(df_temp)


# ****************************************************************************
#  MEN SLOW
# ****************************************************************************
# men slow
df_modeling = df_m_slow.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
dtree = DecisionTreeRegressor()
rfe = RFE(lm,2)
fit = rfe.fit(X_train,y_train) 
lm_predictions = fit.predict(X_test)


res_men_slow = list(compress(X.columns, fit.support_))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

m_slow_list = ['decimaltime',  'szn avg if not last szn avg']
df_modeling = df_modeling[m_slow_list]
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# X_trainC = sm.add_constant(X_train)
# lr=  sm.OLS(y_train, X_trainC)
# olsres = lr.fit()
# X_testC = sm.add_constant(X_test)
# ols_predictions = olsres.predict(X_testC)
# print(olsres.summary())

lm = LinearRegression()
lm.fit(X_train,y_train) 
lm_predictions = lm.predict(X_test)


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

lm.intercept_
coefficents = lm.coef_
coeff_df = pd.DataFrame(coefficents,X.columns,columns=['Coefficient'])



# ****************************************************************************
#  WOMEN FAST
# ****************************************************************************

# women fast
df_modeling = df_w_fast.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
dtree = DecisionTreeRegressor()
rfe = RFE(lm,3)
fit = rfe.fit(X_train,y_train) 
lm_predictions = fit.predict(X_test)

res_women_fast = list(compress(X.columns, fit.support_))

print('MAE:', metrics.mean_absolute_error(y_test, lm_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

df_modeling = df_w_fast.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
w_fast_list = ['decimaltime',  'pr', 'avg previous races','szn avg']
df_modeling = df_modeling[w_fast_list]
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


X_trainC = sm.add_constant(X_train)
lr=  sm.OLS(y_train, X_trainC)
olsres = lr.fit()
X_testC = sm.add_constant(X_test)
ols_predictions = olsres.predict(X_testC)
print(olsres.summary())

lm = LinearRegression()
lm.fit(X_train,y_train) 
lm_predictions = lm.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, lm_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

lm.intercept_
coefficents = lm.coef_
coeff_df = pd.DataFrame(coefficents,X.columns,columns=['Coefficient'])

df_predicted1 = X_test
df_predicted1['decimal time'] = y_test
df_predicted1['predicted'] = lm_predictions
# residual is actual minus predicted. Positive residual means we predictions are low
df_predicted1['residual'] =  df_predicted1['decimal time'] - df_predicted1['predicted']
df_temp = df_w_fast[['year', 'start','name', 'gender', 'school', 'hayward handicap', 'state']]
df_predicted1 = df_predicted1.join(df_temp)
df_predicted_new = pd.concat([df_predicted1, df_predicted])


def approx_rq(time, gender):
       if gender == 'Women':
              if time < 16.51:
                     return 1
              else:
                     return 0
       if gender == 'Men':
              if time < 14.16:
                     return 1
              else:
                     return 0

df_predicted_new['likely rq'] = df_predicted_new.swifter.apply(lambda row: approx_rq(row['decimal time'], row['gender']), axis=1)


def accuracy_approx_rq(time, gender, predicted):
       if gender == 'Women':
              if (time < 16.51) & (predicted == 1):
                     return 1
              elif (time > 16.51) & (predicted == 0):
                     return 1
              else:
                     return 0
       if gender == 'Men':
              if (time < 14.16) & (predicted == 1):
                     return 1
              elif (time > 14.16) & (predicted == 0):
                     return 1
              else:
                     return 0

df_predicted_new['accuracy likely rq'] = df_predicted_new.swifter.apply(lambda row: accuracy_approx_rq(row['predicted'], row['gender'], row['likely rq']), axis=1)

print(confusion_matrix(df_predicted_new['likely rq'],df_predicted_new['accuracy likely rq']))
print(classification_report(df_predicted_new['likely rq'],df_predicted_new['accuracy likely rq']))
print(rfc.score(X_test,y_test)) # t



df_predicted_new.to_excel('12_09_prediction_results.xlsx')







df_baylor_predicted = df_predicted[df_predicted['school'] == 'Baylor']



sns.scatterplot(x='pr' ,y='decimal time', data= df_predicted, hue='state')
# see if I can use hue_norm factor to normalize it
# do different residual plots of residual on the y and various x's on the y

sns.scatterplot(x='decimal time' ,y='residual', data= df_predicted)
# 
# ****************************************************************************
#  WOMEN SLOW
# ****************************************************************************
# women slow

df_modeling = df_w_slow.drop(drop_list, axis = 1)
df_modeling = df_modeling.dropna()
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
dtree = DecisionTreeRegressor()
rfe = RFE(lm,8)
fit = rfe.fit(X_train,y_train) 
lm_predictions = fit.predict(X_test)

res_women_slow = list(compress(X.columns, fit.support_))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))


w_slow_list_best = ['decimaltime', 'pr', 'szn pr', 'szn pr if not last szn pr', 'avg previous races', 'szn avg', 'szn avg handicaped', 'szn avg if not last szn avg', 'interacctionpr*szn pr']
w_slow_list1 = ['decimaltime', 'pr', 'szn pr if not last szn pr', 'szn avg if not last szn avg']
df_modeling = df_modeling[w_slow_list1]
X = df_modeling.drop('decimaltime', axis = 1)
y = df_modeling['decimaltime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


X_trainC = sm.add_constant(X_train)
lr=  sm.OLS(y_train, X_trainC)
olsres = lr.fit()
X_testC = sm.add_constant(X_test)
ols_predictions = olsres.predict(X_testC)
print(olsres.summary())

lm = LinearRegression()
lm.fit(X_train,y_train) 
lm_predictions = lm.predict(X_test)


print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_predictions)))
print('R2:', metrics.r2_score(y_test, lm_predictions))

lm.intercept_
coefficents = lm.coef_
coeff_df = pd.DataFrame(coefficents,X.columns,columns=['Coefficient'])


df2.groupby(['gender', 'state'])['hayward handicap'].first()
df2.groupby('state')['hayward handicap'].first()