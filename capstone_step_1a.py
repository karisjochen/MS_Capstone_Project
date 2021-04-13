
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import date_range
import seaborn as sns
import swifter


df = pd.read_pickle('11_11_clean5k.pkl')
df_avg = pd.read_pickle('11_11_significant_locations.pkl')
df_top5 = pd.read_pickle('11_11_top5_eachlocation.pkl')

# create df for graphing
sig_locations = df_avg[df_avg['sig_location_dummy'] == 1]
location_list = list(sig_locations['location'])
loc_list = location_list+location_list
mens_times = list(sig_locations['Men'])
womens_times = list(sig_locations['Women'])
times = mens_times+womens_times
gender = ['Men']*18 + ['Women']*18
zipped_list = list(zip(loc_list, gender, times))
# use this df for graphying
df_top_sig= pd.DataFrame(zipped_list, columns=['location', 'gender', 'avgtime'])

sns.scatterplot(x='location' ,y='avgtime', data= df_top_sig, hue='gender')
# see if I can use hue_norm factor to normalize it


chart = sns.relplot(x='location',y='avgtime',data=df_top_sig,hue='gender',kind='line')
chart.set_xticklabels(rotation=65, horizontalalignment='right')
chart.savefig('pop_locations.jpeg')



