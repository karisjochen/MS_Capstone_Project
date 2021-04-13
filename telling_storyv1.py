import numpy as np
import pandas as pd
import swifter as swifter
from sklearn.metrics import classification_report, confusion_matrix

# df limited to athletes who have pr offset less than 1.5 minutes
#  92271 men, 72779 women, 165,050 total
df2_m = pd.read_pickle('men_df.pkl')
df2_w = pd.read_pickle('women_df.pkl')

df_rq = pd.read_excel('regional_declarations.xlsx')
df_og = pd.read_excel('12_06_regional_races.xlsx')


kj = df2_w[df2_w['name'] == 'jochen, karis']
kj[['start', 'meet', 'name', 'decimaltime', 'previous number races', 'pr', 'avg previous races', 'days since pr', 'last race',  'szn pr', 'szn avg', 'szn pr if not last szn pr', 'szn avg if not last szn avg']]


dfsig = pd.read_pickle('11_10_significant_locations_top5avg.pkl')

df_times = pd.read_pickle('11_11_significant_locations.pkl')

df_times = df_times.sort_values(by='women handicap')

RQR =['Stanford Invitational', 'Mt. SAC Relays', 'Payton Jordan Invitational']

frames = [df2_w, df2_m]
combined = pd.concat(frames)

meet_groups = combined.groupby('meet')[['push', 'race_size','second place push']].mean()
meet_groups.loc[meet_groups.index == '10th Annual David Suenram Gorilla Classic']
print(meet_groups['push'].mean())
print(meet_groups['second place push'].mean())
print(meet_groups['race_size'].mean())
RQR_groups = meet_groups.loc[meet_groups.index.str.contains('Stanford Invitational|Mt. SAC Relays|Payton Jordan Invitational')]

meet_groups['meet'] = meet_groups.index 

rqrlist = df_og['qualifying meet'].unique()

def find_sig(meet, rqrlist):
    if meet in rqrlist:
        return 1
    else:
        return 0

meet_groups['rqr meet dummy'] = meet_groups.swifter.apply(lambda row: find_sig(row['meet'], rqrlist), axis=1)

print('done')

sig_meet_avg = meet_groups.groupby('rqr meet dummy').mean()

sig_meet_avg

df_predicted = pd.read_excel('12_09_prediction_results.xlsx')


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

df_predicted['predicted likely rq'] = df_predicted.swifter.apply(lambda row: approx_rq(row['predicted'], row['gender']), axis=1)




print(confusion_matrix(df_predicted['likely rq'],df_predicted['predicted likely rq']))
print(classification_report(df_predicted['likely rq'],df_predicted['predicted likely rq']))

df_predicted.to_excel('12_10_prediction_results.xlsx')



