
#from womens2016v2 import df_state_summary
import numpy as np
import datetime
from datetime import date
from numpy.core.multiarray import datetime_as_string
import pandas as pd
import matplotlib.pyplot as plt
from pandas import date_range
import seaborn as sns

import researchpy as rp
import statsmodels.api as sm
import scipy.stats as stats

path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/transformed201609-08_.xlsx'
# this reads all sheets into a dictionary object
all_dfs = pd.read_excel(path, sheet_name=None)

all_dfs.keys()
# put all the sheets into individual dataframes
df = all_dfs['all_athletes']
race_summary = all_dfs['race_summary']
state_summary = all_dfs['state_summary']
regionals_10k = all_dfs['10k_regionals']
regionals_5k = all_dfs['5k_regionals']
d1nationals_regionals = all_dfs['d1nationals_regionals'] # d1 regionals and nationals for both 5k and 10k
d1races_before_regionals = all_dfs['d1races_before_regionals'] # df excluding regionals and d1,2,3 champs
states10k_top50_ncaa = all_dfs['states10k_top50_ncaa']
states5k_top50_ncaa = all_dfs['states5k_top50_ncaa']
topraces10k_top50_ncaa = all_dfs['topraces10k_top50_ncaa']
topraces5k_top50_ncaa= all_dfs['topraces5k_top50_ncaa']


# west_10kQT = regionals_10k.groupby('west_dummy')['qualifying_time'].max()[True]
# east_10kQT = regionals_10k.groupby('west_dummy')['qualifying_time'].max()[False]
# west_5kQT = regionals_5k.groupby('west_dummy')['qualifying_time'].max()[True]
# east_5kQT = regionals_5k.groupby('west_dummy')['qualifying_time'].max()[False]

df2 = df.dropna(subset=['class']) # dropping the pros from the race
regional_qualifiers = df.dropna(subset=['regional_qualifier_10k', 'regional_qualifier_5k'],how='all')
regional_qualifiers_10k = df.dropna(subset=['regional_qualifier_10k'])
regional_qualifiers_5k = df.dropna(subset=['regional_qualifier_5k'])
# 17 girls qualfied for both
regional_qualifiers_both = df.dropna(subset=['regional_qualifier_10k', 'regional_qualifier_5k'],how='any')
df3 = regional_qualifiers_5k
dfcopy = df2



# adding qualifying race stats to all 5k regional qualifiers. May want to add this code to the previous code
qualifiers5k_prechamps = regional_qualifiers_5k[~regional_qualifiers_5k['race_uid'].str.contains(('2016_NCAA Division I Track & Field Championships_5000 Meters_Women'))]
qualifiers5k_prechamps = qualifiers5k_prechamps[~qualifiers5k_prechamps['race_uid'].str.contains(('2016_NCAA Division I Track & Field Championships_10,000 Meters_Women'))]
qualifiers5k_prechamps = qualifiers5k_prechamps.sort_values(['name_fr_url'])
qualifiers5k_prechamps_list = qualifiers5k_prechamps['name_fr_url'].unique()
qtime_list = []
qstate_list = []
num_races_list = []
qplace_list = []
qpush_ratio_list = []

for athlete in qualifiers5k_prechamps_list:
    group = qualifiers5k_prechamps.groupby('name_fr_url').get_group(athlete)
    group.sort_values(['decimaltime'], inplace= True)
    qtime = group.iloc[0].loc['decimaltime']
    qstate = group.iloc[0].loc['state']
    qplace = group.iloc[0].loc['place']
    qpush_ratio = group.iloc[0].loc['push_ratio']
    num_races = len(group)
    qtime_list.append(qtime)
    qstate_list.append(qstate)
    num_races_list.append(num_races)
    qplace_list.append(qplace)
    qpush_ratio_list.append(qpush_ratio)

regionals_5k = regionals_5k.sort_values('name_fr_url')
regionals_5k['qtime'] = qtime_list
regionals_5k['qstate'] = qstate_list
regionals_5k['qplace'] = qplace_list
regionals_5k['qpush_ratio'] = qpush_ratio_list
regionals_5k['races_before_nationals'] = num_races_list
regionals_5k = regionals_5k.drop(columns = ['regional_qualifier_5k','qualifying_time'])


def search_row(place,race): 
    if place < 13 and race == ('2016_NCAA Division I East Preliminary Round_5000 Meters_Women|2016_NCAA Division I West Preliminary Round_5000 Meters_Women'):
        return  1
    else:
        return 0
 
regionals_5k['national_qualifier'] = regionals_5k.apply(lambda row : search_row(row['place'], row['race_uid']), axis = 1)


def search_cali(athlete, df): 
    group = df.groupby('name_fr_url').get_group(athlete)
    if len(group) > 1:
        for index,row in group.iterrows():
            if row['state'] == 'CA':
                return  1
        return 0
    else:
        return 0


dfcopy['cali_girl'] = dfcopy.apply(lambda row : search_cali(row['name_fr_url'], dfcopy), axis = 1)
calidf = dfcopy[dfcopy['cali_girl'] == 1].sort_values(['name_fr_url'])

print('done')

dict_CAfastest = {}
dict_other_statesfastest= {}
dict_CAavg = {}
dict_other_statesavg = {}
cali_df5k = calidf[calidf['distance'] == 5000] # only interested in 5000
cali_athletes5k = cali_df5k['name_fr_url'].unique()
#grouped = calidf.groupby(['name_fr_url','distance', 'state']).sum()
for athlete in cali_athletes5k:
    name = athlete
    group = calidf.groupby(['name_fr_url', 'distance']).get_group((athlete, 5000)) # only interested in 5000
    fastest = group.groupby('state')['decimaltime'].min()
    CAn = 0
    On = 0
    CAsum = 0
    Osum = 0
    for i in range(len(group)):
        if group.iloc[i]['state'] == 'CA':
            CAn +=1
            CAsum += group.iloc[i]['decimaltime']  
        else:
            On +=1
            Osum += group.iloc[i]['decimaltime']
    try:
        CAavg = CAsum/CAn
        fastestCA = fastest[fastest.index == 'CA']
    except:
        CAavg = np.nan
        fastestCA = np.nan
    try:
        Oavg = Osum/On
        fastestother = fastest[fastest.index != 'CA'].min()
    except:
        Oavg = np.nan
        fastestother = np.nan
    dict_CAavg[name] = CAavg
    dict_other_statesavg[name] = Oavg
    dict_CAfastest[name] = fastestCA
    dict_other_statesfastest[name] = fastestother


def add_column(runner, dict): 
    for key, value in dict.items(): 
        if runner == key: 
            return value

    return np.nan

cali_df5k['cali_avg'] = cali_df5k.apply(lambda row : add_column(row['name_fr_url'], dict_CAavg), axis = 1)
cali_df5k['other_states_avg'] = cali_df5k.apply(lambda row : add_column(row['name_fr_url'], dict_other_statesavg), axis = 1)
cali_df5k['CAfastest'] = cali_df5k.apply(lambda row : add_column(row['name_fr_url'], dict_CAfastest), axis = 1)
cali_df5k['other_statesfastest'] = cali_df5k.apply(lambda row : add_column(row['name_fr_url'], dict_other_statesfastest), axis = 1)
cali_df5k['avg_difference'] = cali_df5k['cali_avg'] - cali_df5k['other_states_avg']
cali_df5k['fastest_difference'] = cali_df5k['CAfastest'] - cali_df5k['other_statesfastest']
print('hi ho')

cali_df5k['cali_avg'].corr(cali_df5k['other_states_avg']) # 0.89
cali_df5k['CAfastest'].corr(cali_df5k['other_statesfastest']) # 0.87
cali_df5k['fastest_difference'].corr(cali_df5k['avg_difference']) # 0.73






# today = str(date.today())
# today = today[5:] #just gets the month and date from the string
# filename = '2016_Cali5k_m-dd.xlsx'
# filename = filename.replace('_m-dd', today)
# cali_df5k.to_excel(filename)