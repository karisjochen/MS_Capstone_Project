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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df2_m = pd.read_pickle('men_df.pkl')
df2_w = pd.read_pickle('women_df.pkl')

drop_list_w = ['push ratio', 'second place push ratio', 'decimaltime hayward', 'off szn pr if not last szn pr',
'interaction pr*szn pr', 'interaction pr*avg previous races handicaped', 'interaction pr*szn avg handicaped', 'interaction szn pr*avg previous races handicaped',\
    'interaction szn pr*szn avg handicaped', 'interaction szn pr*last race', 'interaction szn avg handicaped*avg previous races handicaped', 'interaction last race*szn avg handicaped',
    'race_size', 'push', 'second place push', 'sig location', 'sig locations top5avg', 'hayward handicap', 'region', 'state', 'distance', 'interaction pr*last race', 'interaction last race*avg previous races handicaped',
    'szn_pr <= 17.892', 'szn_pr <= 18.625', 'szn_pr <= 20.758', 'szn_pr <= 19.925']
drop_list_m = ['push ratio', 'second place push ratio', 'decimaltime hayward', 'off szn pr if not last szn pr',
'interaction pr*szn pr', 'interaction pr*avg previous races handicaped', 'interaction pr*szn avg handicaped', 'interaction szn pr*avg previous races handicaped',\
    'interaction szn pr*szn avg handicaped', 'interaction szn pr*last race', 'interaction szn avg handicaped*avg previous races handicaped', 'interaction last race*szn avg handicaped',
    'race_size', 'push', 'second place push', 'sig location', 'sig locations top5avg', 'hayward handicap', 'region', 'state', 'distance',
    'interaction pr*last race', 'interaction last race*avg previous races handicaped',
    'szn_pr <= 17.292', 'pr <= 15.225', 'pr <= 14.658', 'pr <= 15.692', 'last_race <= 16.758']


df3_m = df2_m.drop(drop_list_m, axis = 1)
df3_w = df2_w.drop(drop_list_w, axis = 1)



# 20 races, 809 records
df_regionals_w = df3_w[df3_w['race_uid'].str.contains(('NCAA Division I East Preliminary Round|NCAA Division I West Preliminary Round|NCAA Division I East Region|NCAA Division I West Region|NCAA East Preliminary Round|NCAA West Preliminary Round'))]
df_regionals_w['west dummy'] = df_regionals_w['race_uid'].str.contains('West') == True
df_regionals_w['east dummy'] = df_regionals_w['race_uid'].str.contains('East') == True
df_regionals_m = df3_m[df3_m['race_uid'].str.contains(('NCAA Division I East Preliminary Round|NCAA Division I West Preliminary Round|NCAA Division I East Region|NCAA Division I West Region|NCAA East Preliminary Round|NCAA West Preliminary Round'))]
df_regionals_m['west dummy'] = df_regionals_m['race_uid'].str.contains('West') == True
df_regionals_m['east dummy'] = df_regionals_m['race_uid'].str.contains('East') == True

def region(row):
    if row['west dummy'] == 1:
        return "West"
    else:
        return 'East'

df_regionals_w['region'] = df_regionals_w.swifter.apply(lambda row: region(row), axis=1)
df_regionals_m['region'] = df_regionals_m.swifter.apply(lambda row: region(row), axis=1)

print('hi')

def national_q(place):
    if place < 13:
        return 1
    else:
        return 0
    
df_regionals_w['national_qualifiers'] = df_regionals_w.swifter.apply(lambda row: national_q(row['place']), axis = 1)
df_regionals_m['national_qualifiers'] = df_regionals_m.swifter.apply(lambda row: national_q(row['place']), axis = 1)

#nq_list = df_regionals_w[df_regionals_w['national_qualifiers'] == 1].index.tolist()
#missing 6th place from east for 2012 and 2016 and 10th place from east from 2019
df_regionals_w[(df_regionals_w['east dummy'] == True) & (df_regionals_w['year'] == 2019)].sort_values('place')
df_regionals_w[(df_regionals_w['east dummy'] == True) & (df_regionals_w['year'] == 2012)].sort_values('place')

def regional_time(row, df):
    year = row['year']
    name = row['name']
    tempdf = df[(df['year'] == year) & (df['name'] == name)]
    qualifying_time = tempdf['decimaltime'].min()
    return qualifying_time

df_regionals_w['qualifying time'] = df_regionals_w.swifter.apply(lambda row: regional_time(row, df3_w), axis=1)
df_regionals_m['qualifying time'] = df_regionals_m.swifter.apply(lambda row: regional_time(row, df3_m), axis=1)

# concern: 67 women rows have qualifying times > 16:45 so I know this is wrong
# only 8.33% of the qualifiers so maybe its okay to not deal with
# 35 men rows have qualifying times > 14:30 so I know this is wrong

df_ks = df_regionals_w[df_regionals_w['name'].str.contains('karissa')]

def regional_state(row, df):
    year = row['year']
    name = row['name']
    tempdf = df[(df['year'] == year) & (df['name'] == name)]
    qualifying_time = tempdf['decimaltime'].min()
    qualifying_state = tempdf[tempdf['decimaltime'] == qualifying_time].iloc[0]['state']
    return qualifying_state

df_regionals_w['qualifying state'] = df_regionals_w.swifter.apply(lambda row: regional_state(row, df2_w), axis=1)
df_regionals_m['qualifying state'] = df_regionals_m.swifter.apply(lambda row: regional_state(row, df2_m), axis=1)

def rq_meet(row, df):
    year = row['year']
    name = row['name']
    tempdf = df[(df['year'] == year) & (df['name'] == name)]
    qualifying_time = tempdf['decimaltime'].min()
    qualifying_meet = tempdf[tempdf['decimaltime'] == qualifying_time].iloc[0]['meet']
    return qualifying_meet

df_regionals_w['qualifying meet'] = df_regionals_w.swifter.apply(lambda row: rq_meet(row, df2_w), axis=1)
df_regionals_m['qualifying meet'] = df_regionals_m.swifter.apply(lambda row: rq_meet(row, df2_m), axis=1)

# stanford
df_regionals_w[df_regionals_w['region'] == 'West']['qualifying meet'].mode()
# stanford
df_regionals_m[df_regionals_m['region'] == 'West']['qualifying meet'].mode()
# stanford
df_regionals_w[df_regionals_w['region'] == 'East']['qualifying meet'].mode()
# stanford
df_regionals_m[df_regionals_m['region'] == 'East']['qualifying meet'].mode()

# concat women and men
frames = [df_regionals_w, df_regionals_m]
df_regionals = pd.concat(frames)

df_regionals.to_excel('12_06_regional_races.xlsx')
df_regionals.to_pickle('12_06_regional_races.pkl')

df_regionals = pd.read_pickle('12_04_regional_races.pkl')


drop_list_w = ['push ratio', 'second place push ratio', 'decimaltime hayward', 'off szn pr if not last szn pr',
'interaction pr*szn pr', 'interaction pr*avg previous races handicaped', 'interaction pr*szn avg handicaped', 'interaction szn pr*avg previous races handicaped',\
    'interaction szn pr*szn avg handicaped', 'interaction szn pr*last race', 'interaction szn avg handicaped*avg previous races handicaped', 'interaction last race*szn avg handicaped',
    'race_size', 'push', 'region', 'second place push', 'sig location', 'sig locations top5avg', 'hayward handicap',  'distance', 'interaction pr*last race', 'interaction last race*avg previous races handicaped',
    'szn_pr <= 17.892', 'szn_pr <= 18.625', 'szn_pr <= 20.758', 'szn_pr <= 19.925']
drop_list_m = ['push ratio', 'second place push ratio', 'decimaltime hayward', 'off szn pr if not last szn pr',
'interaction pr*szn pr', 'interaction pr*avg previous races handicaped', 'interaction pr*szn avg handicaped', 'interaction szn pr*avg previous races handicaped',\
    'interaction szn pr*szn avg handicaped', 'interaction szn pr*last race', 'interaction szn avg handicaped*avg previous races handicaped', 'interaction last race*szn avg handicaped',
    'race_size', 'push', 'second place push', 'sig location', 'sig locations top5avg', 'hayward handicap',  'distance',
    'interaction pr*last race', 'interaction last race*avg previous races handicaped', 'region',
    'szn_pr <= 17.292', 'pr <= 15.225', 'pr <= 14.658', 'pr <= 15.692', 'last_race <= 16.758']


df4_m = df2_m.drop(drop_list_m, axis = 1)
df4_w = df2_w.drop(drop_list_w, axis = 1)

frames = [df4_w, df4_m]
combined = pd.concat(frames)
region = df_regionals['region']
df4_combined = pd.concat([combined, region],axis=1)

df4_combined.to_excel('12_04_combined_data.xlsx')
df4_combined.to_pickle('12_04_combined_data.pkl')





df_regionals_w = df_regionals[df_regionals['gender'] == 'Women']
# 5 regional qualifiers
df_regionals_w[df_regionals_w['school'].str.contains('Baylor')]

df_w = df4_combined[df4_combined['gender']== 'Women']
df_baylor = df_w[df_w['school'] == 'Baylor']

df_baylor[df_baylor['name'].str.contains('montoya')]

# maggie finished 15th place at regionals her sophomore year but she isnt in my dataset
# her junior year (2016) she started regionals but dnfed so that race isnt in the results either
df_regionals_w[(df_regionals_w['year'] == 2015) & (df_regionals_w['region'] == 'West')]