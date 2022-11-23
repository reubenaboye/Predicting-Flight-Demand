##########################
#title: Predicting Flight Demand - Regression Models
######################

import psycopg2
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

# I. PRELIM ================================================================

# I.1. IMPORT DATA ---------------------------------------------------------------------
fl_train = pd.read_csv("flight_train_ready.csv")
fl_test = pd.read_csv("flight_test_ready.csv")

df = flights_train

# II. DATA CLEANING ==================================================

#       1. Check & Impute Null Values
def null_values_perc(df):
    total_null = df.isnull().sum().sort_values(ascending=False)
    percent_null = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 
    null_df = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])
    return null_df
null_values_perc(df)

def fill_null_val_mean(df, df_col):
    """
    Fills null values in a column with the column's mean value. 
    Note: dataframe column (df_col) must be of numeric type
    """
    col_mean = df[df_col].mean()
    df[df_col].fillna(col_mean, inplace=True)
    return print("Number of Nulls left:",df[df_col].isnull().sum())
fill_null_val_mean(df, 'dep_delay')
fill_null_val_mean(df, 'arr_delay')


def fill_null_val_mode(df, df_col):
    """
    Fills null values in a column with the column's mode value. 
    """
    col_mode = df[df_col].mode()[0]
    df[df_col].fillna(col_mode, inplace=True)
    return print("Number of Nulls left:",df[df_col].isnull().sum())

fill_null_val_mode(df, 'tail_num')



# III. FEATURE ENGINEERING =====================================================

#   III.1. FEATURE CONSTRUCTION

#          1. Date & Time Variables
from datetime import datetime
flight_date = []
for date in df['fl_date']:
    flight_date.append(datetime.strptime(date, "%Y-%m-%d"))

date = pd.Series(flight_date)
df1 = df.merge(date.rename("flight_date"), left_index=True, right_index=True)
df1 = df1.drop('fl_date', axis=1)

fl_years = []
for date in df1['flight_date']:
    fl_years.append(date.year)
years = pd.Series(fl_years)
df1 = df1.merge(years.rename("Year"), left_index=True, right_index=True)

fl_month = []
for date in df1['flight_date']:
    fl_month.append(date.month)
month = pd.Series(fl_month)
df1 = df1.merge(month.rename("Month"), left_index=True, right_index=True)

fl_days = []
for date in df1['flight_date']:
    fl_days.append(date.day)
days = pd.Series(fl_days)
df1 = df1.merge(days.rename("Days"), left_index=True, right_index=True)
    # drop flight date
df1 = df1.drop('flight_date', axis=1)

#       1.2. COnstruct Features: Time of Day
df1['hr_dep'] = df1['crs_dep_time']//100 

time_of_day = []
for hour in df1['hr_dep']:
    if hour < 12:
        time_of_day.append("Morning")
    if hour >=12 and hour < 17:
        time_of_day.append("Afternoon")
    if hour >=17 and hour <= 24:
        time_of_day.append("Evening")
time = pd.Series(time_of_day)
df1 = df1.merge(time.rename('Time of Day'), left_index=True, right_index=True)

#     2. Construct Features: Season
seasons = []
for month in df1['Month']:
    if month <=2 or month > 11:
        seasons.append("Winter")
    if month >= 3 and month <= 5:
        seasons.append('Spring')
    if month >=6 and month <9:
        seasons.append('Summer')
    if month >= 9 and month <= 11:
        seasons.append('Fall')
season_series = pd.Series(seasons)
df1 = df1.merge(season_series.rename("Seasons"), left_index=True, right_index=True)


#      3. Construct Features: Dummy Variables for Categorical Variables

# Variable: Origin
origin_dum = pd.get_dummies(df1['origin'])
# Variable: Destination
dest_dum = pd.get_dummies(df1['dest'])
# Variabel: Mkt Carrier
mktcarrier_dum = pd.get_dummies(df1['mkt_unique_carrier'])
# Variable: Oper. Carrier
opercarrier_dum = pd.get_dummies(df1['op_unique_carrier'])
# Variable: Time of Day
timeofday_dum = pd.get_dummies(df1['Time of Day'])
# Variable: Seasons
season_dum = pd.get_dummies(df1['Seasons'])
# Variable: Tail_num
tail_dum = pd.get_dummies(df1['tail_num'])

# MERGE DUMMIES TO DF
df2 = pd.concat([df1, origin_dum, dest_dum, mktcarrier_dum, opercarrier_dum, 
timeofday_dum, season_dum, tail_dum], axis=1)
#Drop Correpsondent Non-Dummy Ver. of Variables

df2 = df2.drop(['origin', 'dest', 'mkt_unique_carrier', 'op_unique_carrier', 'Time of Day', 'Seasons', 'tail_num'], axis=1)

#   4.  Drop Hypothesized Non-Central Variables
df2 = df2.drop(['origin_city_name', 'dest_city_name', 'origin_airport_id', 'mkt_carrier', 'dest_airport_id', 'dep_delay'], axis=1)


# IV. VARIABLE SELECTION ============================================================================================

#  1. Check for Outliers & Variable Transformation

y = df2['arr_delay']
df3 = df2.drop('arr_delay', axis=1)
df3_numeric = df3.select_dtypes(include='number')

#       1. Feature Selection on Small Variance

from sklearn.feature_selection import VarianceThreshold
vt = VarianceThreshold(0.1)
df3_transformed = vt.fit_transform(df3)

# columns we have selected
# get_support() is method of VarianceThreshold and stores boolean of each variable in the numpy array.
selected_columns = df3.columns[vt.get_support()]
# transforming an array back to a data-frame preserves column labels
df3_transformed_df = pd.DataFrame(df3_transformed, columns = selected_columns)
df3_transformed_df.head() # data with variance above 0.1
    # KEY: we have reduced the features from 798 to 24

df4 = df3_transformed_df.drop(['crs_dep_time'], axis=1)


#       2. Forward Regression/Selection

from sklearn.feature_selection import f_regression, SelectKBest
skb = SelectKBest(f_regression, k=10)
X = skb.fit_transform(df4, y)

skb.get_support()
#column names
df4.columns[skb.get_support()]
X2 = pd.DataFrame(X, columns = df4.columns[skb.get_support()])
X2
# NOW WER ARE DOWN to 10 Columns




# V. MODELLING =======================================================================================
# V.1. PRELIM: Train-Validation-Test Split
from sklearn.model_selection import train_test_split
# set aside 20% of train and test data for evaluation			
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, 
shuffle = True, random_state = 8)

# Use the same function above for the validation set		
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
test_size=0.25, random_state= 8) # 0.25 x 0.8 = 0.2



# V.2. LINEAR REGRESSION --------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
LinearRegression())

pipe_lr.fit(X_train, y_train)
y_train_pred = pipe_lr.predict(X_train)
y_val_pred = pipe_lr.predict(X_val)

# Model Evaluation
from sklearn.metrics import r2_score
print('R^2 train: %.3f, validation: %.3f' %
(r2_score(y_train, y_train_pred),
r2_score(y_val, y_val_pred)))
#R^2 train: 0.015, validation: 0.012
print('MSE train: %.3f, test: %.3f' %(
    mean_squared_error(y_train,y_train_pred),
    mean_squared_error(y_val, y_val_pred)
))
# MSE train: 2448.770, test: 2567.939

#print('Test Accuracy %.3f' % pipe_lr.score(X_val, y_val))
    # Test accuract is 0.012
        # implication: we are underfitting
        # sol'n: More flexible model model like random forest. 

# V.3. RANDOM FOREST ----------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=5)
rfr.fit(X_train, y_train)
y_train_pred_rf = rfr.predict(X_train)
y_val_pred_rf = rfr.predict(X_val)

# Model Evaluation
print('R^2 train: %.3f, validation: %.3f' %
(r2_score(y_train, y_train_pred_rf),
r2_score(y_val, y_val_pred_rf)))