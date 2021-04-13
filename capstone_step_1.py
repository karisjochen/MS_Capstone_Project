import numpy as np
from pandas import date_range
import datetime
from datetime import date
from numpy.core.multiarray import datetime_as_string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import swifter

#lists used in my data import
test = np.arange(96)
class_na = ['Blank', '-', '--', 'en', 'GR', 'NA', 'RE', 'UN']
originaltime_na = ['NT', 'DQ', 'DNS', 'DNF']
place_na = 'Blank'
date_columns = [1,2,3]
dict_na = {'Class': test, 'OriginalTime': originaltime_na, 'Place': place_na}

path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/data/tffrs_raw_kdj.xlsx'
df = pd.read_excel(path, sheet_name = 'resultskdj', header = 1, index_col = 0,skiprows = 0)



df = pd.read_excel(path, sheet_name = 'resultskdj', header = 1, index_col = 0, na_values = dict_na,\
    keep_default_na = True, skiprows = 0, parse_dates = date_columns)


df.columns = df.columns.str.lower()
df['name'] = df['name'].str.lower()

#create year column

df['year'] = pd.DatetimeIndex(df['start']).year
df['year'].head()

df.dropna(subset=['originaltime'], inplace = True)


#further cleanup of Class column, inplace = True is necessary
#to change original df


na_list = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
df['class'].replace(class_na, na_list, inplace=True)


df.drop_duplicates(subset = ['athlete_id', 'distance', 'start', 'originaltime'], inplace=True)


df['originaltime'] = df['originaltime'].astype(str)
H_list = df[df['originaltime'].str.contains('H')].index.tolist()
df.drop(H_list, inplace = True)


#setting up datetime format of original time column. this will allow for subtractions
def format(x):
    try:
        at_index_dot = x.find('.')
        at_index_colon = x.find(':')
        x = x[at_index_colon+1: at_index_dot]


        FMT = '%M:%S'
        #FMT = '%M:%S:f' #microsecond as a decimal number
        x = datetime.datetime.strptime(x, FMT)
 
        return x
    except:
        return 'Nan'

    
df['originaltime'] = df['originaltime'].apply(lambda x: format(x))
na_list = df[df['originaltime'] == np.nan].index.tolist()
df.drop(na_list, inplace = True)


df['race_uid'] = df['year'].astype(str)+'_'+df['meet']+'_'+df['event']+'_'+df['gender']
print('hi, hello')

#create decimal time
def decimaltime(x):
    try:
        strtime = str(x)
        strtime = strtime[14:] #15:57
        at_index_colon = strtime.find(':')
        minutes = strtime[:at_index_colon]
        seconds = strtime[at_index_colon+1:]
        minutes = int(minutes)
        seconds = int(seconds)
        total_seconds = minutes*60 + seconds
        decimaltime = total_seconds/60
        return decimaltime

    except:
        return np.nan

    
df['decimaltime'] = df['originaltime'].swifter.apply(lambda x: decimaltime(x))

#extract state from each location and add df column

states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
def extractstate(x):
    try:

        for i in states:
            if i in x:
                state = i
                return state
                break
    except:
        return np.nan



df['state'] = df['location'].apply(lambda x: extractstate(x))



states = {
        'AK': 'Other',
        'AL': 'South',
        'AR': 'South',
        'AS': 'Other',
        'AZ': 'West',
        'CA': 'California',
        'CO': 'West',
        'CT': 'North',
        'DC': 'North',
        'DE': 'North',
        'FL': 'South',
        'GA': 'South',
        'GU': 'Other',
        'HI': 'Hawaii',
        'IA': 'Mid-West',
        'ID': 'West',
        'IL': 'Mid-West',
        'IN': 'Mid-West',
        'KS': 'Mid-West',
        'KY': 'South',
        'LA': 'South',
        'MA': 'North',
        'MD': 'North',
        'ME': 'North',
        'MI': 'West',
        'MN': 'Mid-West',
        'MO': 'Mid-West',
        'MP': 'Other',
        'MS': 'South',
        'MT': 'West',
        'NA': 'Other',
        'NC': 'South',
        'ND': 'Mid-West',
        'NE': 'West',
        'NH': 'North',
        'NJ': 'North',
        'NM': 'West',
        'NV': 'West',
        'NY': 'North',
        'OH': 'Mid-West',
        'OK': 'South',
        'OR': 'Oregon',
        'PA': 'North',
        'PR': 'Other',
        'RI': 'North',
        'SC': 'South',
        'SD': 'Mid-West',
        'TN': 'South',
        'TX': 'South',
        'UT': 'West',
        'VA': 'South',
        'VI': 'Other',
        'VT': 'North',
        'WA': 'West',
        'WI': 'Mid-West',
        'WV': 'South',
        'WY': 'West'
}

df['region'] = df['state'].map(states)


# further clean up class column

df['class'].replace({'FR-1': 'Freshman', 'SO-2': 'Sophomore', 'SR-4': 'Senior', 'JR-3': 'Junior'}, inplace = True)
df['class'].replace({'FR': 'Freshman', 'SO': 'Sophomore', 'SR': 'Senior', 'JR': 'Junior'}, inplace = True)
df['class'].replace({'Fr': 'Freshman', 'So': 'Sophomore', 'Sr': 'Senior', 'Jr': 'Junior'}, inplace = True)


df.drop('event', axis=1,inplace=True)
df.drop('name_fr_url', axis=1,inplace=True)
df.drop(na_list, inplace = True)

# this helps save memory
df['state'] = df['state'].astype('category')
df['year'] = df['year'].astype('category')
df['gender'] = df['gender'].astype('category')
df['distance'] = df['distance'].astype('category')
df['class'] = df['class'].astype('category')
df['region'] = df['region'].astype('category')
df.dropna(subset=['decimaltime'], inplace=True)
df.dropna(subset=['state'], inplace=True)



# in a df of 314,725 entries (10k and 5k) there are 786 locations
df['location'].value_counts()

# limiting down to 5ks now we have 247,177 entries and 781 locations
df2 = df[df['distance'] == 5000]
df2['location'].value_counts()

# 86 locations in CA
df3 = df2[df2['state'] == 'CA']
df3['location'].value_counts()

# 19 locations in OR
# 'Hayward Field - Eugene, OR' is our base
df4 = df2[df2['state'] == 'OR']
df4['location'].value_counts()

# we only care about people who can run a 5k below 24 minutes
# 12.616 is decimal time WR for men 2005
# 14.183 is decimal time WR for women 2008
df2 = df2[df2['decimaltime'] < 24]
df2 = df2[df2['decimaltime'] > 13]
sns.scatterplot(x=df2['location'],y=df2['decimaltime'], hue=df2['year'])

df2.describe()



# find all of the mt. sac locations
df2['state'] = df2['state'].astype('str')
df2[(df2['meet'].str.contains('SAC')) & (df2['state'] == 'CA')]['location'].value_counts()
'Cerritos - Norwalk, CA' # this is mt. sac
'El Camino College - Torrance, CA'
'Mt. San Antonio College-Walnut, CA - Walnut, CA'
'Westmont College - Santa Barbara, CA' 
df2.loc[(df2['meet'].str.contains('SAC')) & (df2['state'] == 'CA'), 'location'] = 'Mt. SAC Relays'
df2[(df2['meet'].str.contains('SAC')) & (df2['state'] == 'CA')]['location']

top_locations = df2.groupby('location').size()

# the locations are now the index of a pd Series. Below line searches
# whether a location contains a specific phrase. For ex, TAMU is put down
# as two different locations
top_locations[top_locations.index.str.contains('College Station, TX')]
dict_replace = {'- College Station, TX':'Texas A&M - College Station, TX', 'Stanford - Palo Alto, CA': 'Stanford Cobb Track and Angell Field - Palo Alto, CA',\
    'Sacramento - Sacramento, CA': 'Sacramento St. - Sacramento, CA', 'Sacramento State, CA': 'Sacramento St. - Sacramento, CA',\
    'University of Texas (Eastern Deadline) - Austin, TX': 'Texas-Mike A. Myers Stadium - Austin, TX', 'Texas A&M-Kingsville (Eastern Deadline) - Kingsville, TX':\
    'Texas A&M-Kingsville - Kingsville, TX'}

top_locations[top_locations.index.str.contains('Hayward')] # i think this looks fine
top_locations[top_locations.index.str.contains('Palo Alto')] # need to replace some
top_locations[top_locations.index.str.contains('SAC')] # 2889 just like we want!
top_locations[top_locations.index.str.contains('Texas')] # made some assumptions here
top_locations[top_locations.index.str.contains('John McDonnell Field')] # Arkansas location only 272 races, less than I thought
'Arkansas-John McDonnell Field - Fayetteville, AR'


df2['location'].replace(dict_replace,inplace=True)
df2.drop('event', axis=1,inplace=True)

# clean 5k data
df2.to_pickle('11_11_clean5k.pkl')



# top 100 locations
top_locations2 = top_locations[top_locations > 620]
top_locations2.sum()

top_locations_list = list(top_locations2.index)
# df of races at one of the top 99 locations, 119,356 values
df_top_locations = df2[df2['location'].isin(top_locations_list)]

def identify_top_location(location, region, list):
    if location in list:
        return location
    else:
        return region

df2['sig location'] = df2.swifter.apply(lambda row: identify_top_location(row['location'], row['region'], top_locations_list), axis=1)


# this scatter plot below is madness
sns.scatterplot(x=df2['top_locations'] ,y= df2['decimaltime'], hue=df2['region'] )

df_avg_time = df2.groupby('top_locations')['decimaltime'].mean()
df_avg_time.sort_values(ascending=True)

df_top_boygirl = df2.groupby(['top_locations', 'gender'])
df_top_boygirl = df_top_boygirl.apply(lambda x: x.sort_values(['decimaltime'], ascending=True))
df_top_boygirl = df_top_boygirl.reset_index(drop=True)
# 100 locations plus 7 regions = 107 locations x 10 athletes = 1070 values
df_top_boygirl = df_top_boygirl.groupby(['top_locations', 'gender']).head(5)

# df_top_boygirl.to_pickle('11_11_top5_eachlocation.pkl')


df_avg_timetop_boygirl = df_top_boygirl.groupby(['top_locations', 'gender'])['decimaltime'].mean()
df_avg_timetop_boygirl = df_avg_timetop_boygirl.sort_values(ascending=True)
df_avg_timetop_boygirl  = df_avg_timetop_boygirl.unstack(level=1)



test = dict(zip(df_avg_timetop_boygirl.index,df_avg_timetop_boygirl.values ))

# df_avg_timetop_boygirl.to_excel('top5_eachlocation_avgtime.xlsx')

# 5 fastest and 5 slowest when men and women combined

df_avg_timetop_boygirl.sort_values(by='Men', ascending=True)
men_fastest = ['Stanford Cobb Track and Angell Field - Palo Alto, CA', 'Mt. SAC Relays',\
    'Sacramento St. - Sacramento, CA',\
    'Azusa Pacific (O) - Azusa, CA','Hayward Field - Eugene, OR']
men_slowest = ['Westminster (Pa.) - New Wilmington, PA', 'West Chester - West Chester, PA',\
    'Lenoir-Rhyne - Hickory, NC', 'Benedictine University - Lisle, IL', 'Incarnate Word - San Antonio, TX']
df_avg_timetop_boygirl.sort_values(by='Women', ascending=True)
women_fastest = ['North']
women_slowest = ['Westminster (Pa.) - New Wilmington, PA', 'Rose-Hulman - Terre Haute, IN',\
    'Rochester - Rochester, NY', 'Birmingham-Southern - Birmingham, AL']
significant_locations = men_fastest + men_slowest + women_fastest + women_slowest

# creating above summary table into a dataframe I can work with
location_index = pd.Series(df_avg_timetop_boygirl.index)
location_index_list = list(location_index)
boy_times = df_avg_timetop_boygirl['Men']
girl_times = df_avg_timetop_boygirl['Women']
zipped_list = list(zip(location_index_list, boy_times, girl_times))
df_times = pd.DataFrame(zipped_list, columns=['location', 'Men', 'Women'])

def identify_sig_locations(location,list):
    if location in list:
        return 1
    else:
        return 0

df_times['graph_location_dummy'] = df_times.swifter.apply(lambda row: identify_sig_locations(row['location'],significant_locations), axis=1)

# df_times.to_excel('11_10_significant_locations_top5avg.xlsx')
# df_times.to_pickle('11_11_significant_locations.pkl')

def input_sig_location(location, gender, df):
    try:
        return df[df['location'] == location][gender].iloc[0]
    except:
        return np.nan

df2['sig locations top5avg'] = df2.swifter.apply(lambda row: input_sig_location(row['sig location'], row['gender'], df_times), axis=1)

df_times['gender avg'] = ((df_times['Men'] + df_times['Women'])/2)
df_times.sort_values('gender avg', ascending=True)
w_hayward = 15.333333
df_times['women handicap'] = (df_times['Women']/w_hayward)*100
m_hayward = 13.376667
df_times['men handicap'] = (df_times['Men']/m_hayward)*100
df_times['genderavg hayward handicap'] = ((df_times['gender avg'])/((m_hayward+w_hayward)/2))*100


def input_handicap(location, gender, df):
    if gender == 'Women':
        return df[df['location'] == location]['women handicap'].iloc[0]
    if gender == 'Men':
        return df[df['location'] == location]['men handicap'].iloc[0]

df2['hayward handicap'] = df2.swifter.apply(lambda row: input_handicap(row['sig location'],row['gender'],df_times), axis=1)

df2.drop(['sport', 'state_prov', 'end'], axis=1, inplace=True)
#df2.to_pickle('11_11_clean5k.pkl')
#df_times.to_pickle('11_11_significant_locations.pkl')


