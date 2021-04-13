

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





# using this dataframe for running models

path2 = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/11_07_full_cali_full_features.pkl'
dfcali = pd.read_pickle(path2)
dfcali['push'] = dfcali['push'].fillna(0)
dfcali['second_place_push'] = dfcali['second_place_push'].fillna(0)
def competitive_factor(pr):
    if pr <= 13:
        return 'less than or equal to 13'
    if 13> pr <= 14.5:
        return 'between 13 and 14.5 minutes'
    if 14.5> pr <= 16:
        return 'between 14.5and 16 minutes'
    if 16> pr <= 17.5:
        return 'between 16 and 17.5 minutes'
    if 17.5> pr <= 19:
        return 'between 17.5 and 19 minutes'
    if 19> pr <= 20.5:
        return 'between 19 and 20.5 minutes'
    else:
        return 'greater than 20.5 minutes'
dfcali['competitive_factor'] = dfcali.swifter.apply(lambda row: competitive_factor(row['previous best time']), axis=1) 

sns.distplot(dfcali['previous best time'], bins=7)
sns.scatterplot()
# ****************************************************************************
#  PREDICTING ALL RACES IN ANY LOCATION MODEL LINEAR REGRESSION AFTER DEFINING CONSTANT
# ****************************************************************************

df_modeling2 = dfcali[dfcali['state'] == 'CA']
df_modeling2 = df_modeling2[['decimaltime', 'gender','race_size', 'competitive_factor', 'second_place_push',  'average of previous races','previous best time', 'previous number races', 'days since previous best time', 'previous best region']]

df_modeling2 = df_modeling2.dropna() # drop the first race of the season bc we are not trying to predict times here, should have 42,751 rows
cat_features2 = ['previous best region',  'gender', 'competitive_factor']
dummy_df2 = pd.get_dummies(df_modeling2, columns=cat_features2, drop_first=True)

y2 = dummy_df2['decimaltime']
X2 = dummy_df2.drop(['decimaltime'], axis = 1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=101)

X_trainC2 = sm.add_constant(X_train2)
lr2 = sm.OLS(y_train2, X_trainC2)
olsres2 = lr2.fit()
X_testC2 = sm.add_constant(X_test2)
ols_predictions = olsres2.predict(X_testC2)
print(olsres2.summary())

sns.distplot((y_test2-ols_predictions),bins=50)

# linear regression using sci-kit learn
lm2 = LinearRegression()
lm2.fit(X_train2,y_train2) 
lm_predictions2 = lm2.predict(X_test2)

# can go back and create a df with the model intercept and coefficients if it looks like it will be meaninful
lm2.intercept_
coefficents2 = lm2.coef_
coeff_df2 = pd.DataFrame(lm2.coef_,X2.columns,columns=['Coefficient'])
coeff_df2

df_predicted2 = X_test2
df_predicted2['decimal time'] = y_test2
df_predicted2['predicted decimal time'] = lm_predictions2
df_predicted2['residual'] =  df_predicted2['decimal time'] - df_predicted2['predicted decimal time']
region_df2 = dfcali2[['Region', 'gender']]
df_predicted2 = df_predicted2.join(region_df2)

# now this is the same as calculated RMSE
print('MAE:', metrics.mean_absolute_error(y_test2, lm_predictions2))
print('MSE:', metrics.mean_squared_error(y_test2, lm_predictions2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test2, lm_predictions2)))
print('R2:', metrics.r2_score(y_test2, lm_predictions2)) 


race = df['year'].astype(str)+'_'+df['meet']+'_'+df['event']+'_'+df['gender']
print('hi, hello')


def proxy_size(race, year,df):
    year_replace = year - 1
    year_replace = str(year_replace)
    year = str(year)
    race = race.replace(year,year_replace)
    try:
        return df.loc[race]
    except:
        return np.nan

size_series = pd.Series(dfcali.groupby('race_uid').size())
dfcali['proxy_race_size'] = dfcali.swifter.apply(lambda row : proxy_size(row['race_uid'], row['year'],size_series), axis = 1)


list_c = []
list_v = []
list_v2 = []
constant = lm2.intercept_

for index in coeff_df2.index:
    list_c.append(index)

for value in coeff_df2.values:
    list_v.append(value)
for i in range(len(list_v)):
    list_v2.append(list_v[i][0])

best_model_coeff_dict = dict(zip(list_c,list_v2))
   

test = all_5k.groupby('Region')[['race_size', 'push', 'second_place_push']].mean()
def fill_proxies(row, df):
    region = row['Region']
    for x in df.columns:
        proxy = df.loc[region][x]
        row[x] = proxy
    
    return row

temp_df_predicted2 = df_predicted2
temp_df_predicted2 = temp_df_predicted2.swifter.apply(lambda row: fill_proxies(row, test), axis=1)

def apply_model(row,dict,constant):
    temp_list = list(map(lambda x,y: row[x]*y, dict.keys(), dict.values()))
    return sum(temp_list) + constant

temp_df_predicted2['proxy_predictions'] = temp_df_predicted2.swifter.apply(lambda row : apply_model(row,best_model_coeff_dict, constant), axis = 1)
temp_df_predicted2['proxy predicted residual'] =  temp_df_predicted2['decimal time'] - temp_df_predicted2['proxy_predictions']

avg_proxy_residual = ((temp_df_predicted2['proxy predicted residual']**2).mean())**(1/2)



print('PROXY MAE:', metrics.mean_absolute_error(temp_df_predicted2['decimal time'], temp_df_predicted2['proxy_predictions']))
print('PROXY MSE:', metrics.mean_squared_error(temp_df_predicted2['decimal time'], temp_df_predicted2['proxy_predictions']))
print('PROXY RMSE:', np.sqrt(metrics.mean_squared_error(temp_df_predicted2['decimal time'], temp_df_predicted2['proxy_predictions'])))
print('PROXY R2:', metrics.r2_score(temp_df_predicted2['decimal time'], temp_df_predicted2['proxy_predictions'])) 

