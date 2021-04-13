import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import date_range
import seaborn as sns
from sklearn import metrics # calculate MSE and RMSE for linear regression
import swifter

path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/hopethisisthelasttime_11_04.xlsx'
# this reads all races into df project. WARNING THIS DF HAS 10K IN IT TOO
all_dfs = pd.read_excel(path)
all_5k = all_dfs[all_dfs['distance'] == 5000]
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
        'HI': 'Other',
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
        'WY': 'West',
        np.nan: np.nan
}
all_5k['Region'] = all_5k['state'].map(states)

path2 = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/11_07_full_cali_full_features.pkl'
dfcali2 = pd.read_pickle(path2)

cali_races = all_5k[all_5k['state'] == 'CA']
w_cali_races = cali_races[cali_races['gender'] == 'Women']
m_cali_races = cali_races[cali_races['gender'] == 'Men']


# dataframe has 2000 rows. 100 men, 100 women times in 2009-2019
top_100 = all_5k.groupby(['year', 'gender'])
top_100 = top_100.apply(lambda x: x.sort_values('decimaltime'))
top_100 = top_100.reset_index(drop=True)
top_100 = top_100.groupby(['year', 'gender']).head(100)

state_percent = []

list_states = []
dict_state_percent = {}

for index in top_100['state'].value_counts().index:
    list_states.append(index)

other = 0
for value in top_100['state'].value_counts():
    state_percent.append(value/2000*100)
    if value < 60:
        other = other + value
other = other/2000*100
top6_states = list_states[0:6]
top6_state_percent = state_percent[0:6]
top6_states.append('Other')   
top6_state_percent.append(other)



fig, ax = plt.subplots()
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#909090'
plt.rcParams['axes.labelcolor']= '#909090'
plt.rcParams['xtick.color'] = '#909090'
plt.rcParams['ytick.color'] = '#909090'
plt.rcParams['font.size']=12
figure_title = 'Mens and Womens Top 100 Times each Year by State'
ax.pie(top6_state_percent, labels=top6_states,  
    autopct='%1.0f%%', shadow=True, startangle=0, pctdistance=1.2,labeldistance=1.4)
ax.axis('equal')
ax.set_title(label = figure_title, loc = 'center', y=1.2)
ax.legend(frameon=False, bbox_to_anchor=(1.3,0.8))

plt.savefig('state_percentages.pdf', bbox_inches='tight')
plt.close()