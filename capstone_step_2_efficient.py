
import numpy as np
import datetime
from datetime import date
from numpy.core.arrayprint import _TimelikeFormat
from numpy.core.multiarray import datetime_as_string
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas import date_range
import swifter
from time import  time, process_time


path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/full_data_clean10-31.xlsx'
df3 = pd.read_pickle(path)


start_time = time()
# drop rows that have missing values in either state, decimaltime, name
df3.dropna(subset=['state'], axis=0, inplace=True)
df3.dropna(subset=['decimaltime'],  axis=0, inplace=True)
df3.dropna(subset=['class'],  axis=0, inplace=True)
na_list = df3[df3['name_fr_url'] == 'No Athlete Name fr URL'].index.tolist()
df3.drop(na_list, inplace = True)

df3['decimaltime'] = pd.to_numeric(df3['decimaltime'])
df3['class'].replace({'FR-1': 'Freshman', 'SO-2': 'Sophomore', 'SR-4': 'Senior', 'JR-3': 'Junior'}, inplace = True)

df2 = df3[['race_uid', 'gender', 'start', 'year', 'state', 'class', 'school', 'distance', 'name_fr_url', 'place', 'originaltime', 'decimaltime']]
df3 = df3.sort_values(by=['race_uid', 'decimaltime'])

passed_time = time() - start_time 

print(f"It took {passed_time}")






start_time = time()


def size(race, df):
    return len(df.groupby('race_uid').get_group(race))


def add_column(df): 
    return pd.Series(size(row['race_uid'], df) for index, row in df.iterrows())

df3['size_iterrows'] = add_column(df3)

passed_time = time() - start_time 

print(f"It took {passed_time}") # i think this one took like 8 hours


print('hi')


start_time2 = time()

def size(race, df):
    return len(df.groupby('race_uid').get_group(race))


def add_column(df): 
    return pd.Series(size(row['race_uid'], df) for row in df.itertuples())

df3['size_iterrows'] = add_column(df3)

passed_time2 = time() - start_time 
print(f"It took {passed_time2}")
###############################################################
# first see if making string objects into categorical objects saves memory
# then figure out how to tell time of function
# then work on vecorizing



path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/hopethisisthelasttime_11_04.xlsx'
df = pd.read_excel(path)

df['name_fr_url'] = df['name_fr_url'].str.lower()

# save memory by creating categorical data types

df['state'] = df['state'].astype('category')
df['year'] = df['year'].astype('category')
df['gender'] = df['gender'].astype('category')
df['distance'] = df['distance'].astype('category')
df['class'] = df['class'].astype('category')


# ******************************************************************
# DF OF RUNNERS WHO HAVE 2 OR MORE 5KS AND AT LEAST ONE WAS RUN IN CA
# THIS PROCESS TOOK 17 HOURS. SEE BELOW FOR CODE THAT DOES EXACT SAME
# THING BUT IN LESS THAN A MINUTE
# ******************************************************************


df2 = df[df['distance'] == 5000]

t1_start = process_time()
def search_cali(athlete, df):
    group = df.groupby('name_fr_url').get_group(athlete)
    if len(group) > 1 and 'CA' in group['state'].unique():
        return  1
    else:
        return 0
df2['cali'] = df2.swifter.apply(lambda row : search_cali(row['name_fr_url'], df2), axis = 1)
t1_stop = process_time()
elapsed_time = t1_stop - t1_start
print('cali df elapsed time in seconds {}'.format(elapsed_time))


df2 = df2.sort_values(by='name_fr_url')

# filename = '11_05_fulldata_withCali_dummy.xlsx'
# filename2 = '11_05_fulldata_withCali_dummy.pkl'
# df2.to_excel(filename)
# df2.to_pickle(filename2)
print('hi ho')

# ******************************************************************
# DF OF RUNNERS WHO HAVE 2 OR MORE 5KS AND AT LEAST ONE WAS RUN IN CA
# EFFICIENT CODE THAT TAKES ~1 MINUTE
# ******************************************************************



smdf = df2[['name_fr_url', 'state']]
smdf = smdf.sort_values(by='name_fr_url')
df2 = df2.sort_values(by='name_fr_url')
t1_start = process_time()
def search_cali3(group):
    if 'CA' in group['state'].unique():
        return 1
    else:
        return 0
def search_athlete3(df):
    cali_list = []
    groups = df.groupby('name_fr_url')
    for name, group in groups:
        if len(group) > 1:
            result = search_cali3(group)
            cali_list.append([result]*len(group))
        else:
            result = 0
            cali_list.append([result]*len(group))
    return cali_list

# test is a list of lists    
test  = search_athlete3(smdf)
t1_stop = process_time()
elapsed_time = t1_stop - t1_start
print('cali df elapsed time in seconds {}'.format(elapsed_time))
print('hi ho')

# unflatten the list of lists
# set the list equal to our cali column
flat_list = list(np.concatenate(test))
df2['cali'] = flat_list

cali_df = df2[df2['cali'] == 1]

filename = '11_05_Cali_peeps_only.xlsx'
filename2 = '11_05_Cali_peeps_only.pkl'
cali_df.to_excel(filename)
cali_df.to_pickle(filename2)



y = 1
def test(x,z,y):
    x.append(y)
    z.append(y+1)
    return x,z

x = []
z = []
test(x,z,y)







#####################################################################
# calculating push indicator

push = []
push_ratio = []
for race in races:
    group = df2.groupby('race_uid').get_group(race)
    group = group.reset_index()
    for index,row in group.iterrows():
        if len(group) > 1:
            time1 = group.iloc[0].loc['decimaltime']
            time2 = group.iloc[index].loc['decimaltime']
            push.append(time2 - time1)
            push_ratio.append((time2 - time1)/time1)
        if len(group) == 1:
            push.append(np.nan)
            push_ratio.append(np.nan)

print('ho')

# recalculate second place push difference/first
second_place_push = []
second_place_push_ratio = []
for race in races:
    group = df2.groupby('race_uid').get_group(race)
    if len(group) > 1:
        time1 = group.iloc[0].loc['decimaltime']
        second_place = group.iloc[1].loc['decimaltime']
        second_place_push.append(second_place-time1)
        second_place_push_ratio.append((second_place-time1)/time1)
    if len(group) == 1:
        second_place_push.append(np.nan)
        second_place_push_ratio.append(np.nan)


print('hi ho')

df2['push'] = push
df2['push_ratio'] = push_ratio
df2.astype({'push': 'float64'})

list_second_place_push = []
dict_second_place_push = {races[i]: second_place_push[i] for i in range(len(races))}
list_second_place_push_ratio = []
dict_second_place_push_ratio = {races[i]: second_place_push_ratio[i] for i in range(len(races))}


def get_value(search, my_dict, my_list): 
    for key, value in my_dict.items(): 
         if search == key: 
             my_list.append(value) 
  
    return "key doesn't exist"

for row in df2['race_uid']:
    get_value(row,dict_second_place_push, list_second_place_push)
    get_value(row,dict_second_place_push_ratio, list_second_place_push_ratio)
df2['second_place_push'] = list_second_place_push
df2['second_place_push_ratio'] = list_second_place_push_ratio

print('done. Now make the cali df')

today = str(date.today())
today = today[5:] #just gets the month and date from the string
filename = 'full_data_clean2_m-dd.pkl'
filename = filename.replace('_m-dd', today)
# df.to_excel(filename)
df2.to_pickle(filename)



# ******************************************************************
#  
# OLD METHOD OF DOING IT FOR 2016 DATA
# 
# 
# 
# DF OF RUNNERS WHO HAVE 2 OR MORE 5KS AND AT LEAST ONE WAS RUN IN CA
# ******************************************************************

df2 = df[df['distance'] == 5000]

def search_cali(athlete, df): 
    group = df.groupby('name_fr_url').get_group(athlete)
    if len(group) > 1:
        for index,row in group.iterrows():
            if row['state'] == 'CA':
                return  1
        return 0
    else:
        return 0



df2['cali'] = df2.apply(lambda row : search_cali(row['name_fr_url'], df2), axis = 1)
cali_df5k = df2[df2['cali'] == 1].sort_values(['name_fr_url'])
#cali_df5k = cali_df5k[cali_df5k['distance'] == 5000]


print('hi ho')

dict_CAfastest = {}
dict_other_statesfastest= {}
dict_CAavg = {}
dict_other_statesavg = {}
#cali_df5k = calidf[calidf['distance'] == 5000] # only interested in 5000
cali_athletes5k = cali_df5k['name_fr_url'].unique()
for athlete in cali_athletes5k:
    name = athlete
    group = cali_df5k.groupby(['name_fr_url']).get_group(athlete) # only interested in 5000
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

cali_df5k.dropna(subset=['other_statesfastest', 'other_states_avg'], inplace=True) #drop records of ppl who dont have times outside of cali


print('done')

# ******************************************************************
# CREATING EXCEL FILE FOR TRANSFORMED DATA
# ******************************************************************
today = str(date.today())
today = today[5:] #just gets the month and date from the string
filename = 'transformed2016_m-dd.xlsx'
filename = filename.replace('_m-dd', today)


with pd.ExcelWriter(filename) as writer:  
     df2.to_excel(writer, sheet_name='all_athletes')
     cali_df5k.to_excel(writer, sheet_name='cali_df5k')
     

# df2copy = df.copy()
# append df to existing excel file
# with pd.ExcelWriter('output.xlsx',
#                      mode='a') as writer:  
#      df.to_excel(writer, sheet_name='Sheet_name_3')


print('done')

'''

filename = 'hopethisisthelasttime_11_04.xlsx'
df.to_excel(filename)