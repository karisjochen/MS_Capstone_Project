import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import date_range
import seaborn as sns
from seaborn.regression import lmplot
import swifter

path = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/hopethisisthelasttime_11_04.xlsx'
# this reads all races into df project. WARNING THIS DF HAS 10K IN IT TOO
df = pd.read_excel(path)

path2 = '/Users/Karis/Documents/Data_Analytics/Capstone_Running/11_07_full_cali_full_features.pkl'
dfcali = pd.read_pickle(path2)
dfcali['push'] = dfcali['push'].fillna(0)
dfcali['second_place_push'] = dfcali['second_place_push'].fillna(0)

# ****************************************************************************
#  LOOK AT LOCATION CORRECTION FACTOR
#  HAYWARD FIELD IS OUR BASELINE
#  HOW MANY DIFFERENT LOCATIONS ARE IN CA?
#  START WITH CA, GROUP BY LOCATION? SCATTERPLOT OF Y=TIME, X=LOCATION, HUE=YEAR
# ****************************************************************************

avg_by_location = df.groupby('location')['decimaltime'].mean()


