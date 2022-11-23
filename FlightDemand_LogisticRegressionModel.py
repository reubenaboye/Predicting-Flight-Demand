###########################
# title: Predicting Flight Demand - Logistic Regression Model
#########################

# I. PRELMIINARIES===========================================================================

import psycopg2
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
# 1.1. IMPORT DATA----------------------------------------------------------------------------

fl_train = pd.read_csv("flight_train_ready.csv")
fl_test = pd.read_csv("flight_test_ready.csv")
df = fl_train

# II. DATA EXPLORATION=============================================================================================
col_names = list(df.columns)
for col in col_names:
    print(df[col].value_counts(dropna=False))   

    #check for null values
    #Create series with null value data for out dataframe
total_null = df.isnull().sum().sort_values(ascending=False)
percent_null = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 
    # note here we are dividin the number of null values in each column by the number of observations. 

    #merge data into single dataframe
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent']) # herw we merge the above series into a dataframe
missing_data.head(20)
    # KEY: arr_delay, dep_delay and tail_num have null values
    
# 1.1. Fill NA values 

def fill_null_val_mean (df, df_col):
    """
    Fills null values in a column with the column's mean value. 
    Note: dataframe column (df_col) must be of numeric type
    """
    col_mean = df[df_col].mean()
    df[df_col].fillna(col_mean, inplace=True)
    return print("Number of Nulls left:" ,df[df_col].isnull().sum())
fill_null_val_mean(df, 'dep_delay')
fill_null_val_mean(df, 'arr_delay')

df2= df.drop("dup", axis=1)

df2 = df2.drop('tail_num', axis=1)

# III. FEATURE ENGINEERING==============================================================================================================

# 3.1. Create Target Variable---------------------------------------------------------------------
from sklearn.preprocessing  import LabelEncoder

df2['Delayed_Arrival'] = pd.cut(df2['arr_delay'], bins=[-70, 15, 1400], labels=[0, 1])
df2 = df2.drop(['arr_delay'], axis = 1)
le = LabelEncoder()
le.fit(df2['Delayed_Arrival'])
le.classes_
df2 = df2.dropna(subset=['Delayed_Arrival'])

    # check for class imbalance
print(df2['Delayed_Arrival'].value_counts(normalize=True))

#NOTE: account for imbalances in classifier hyperparameter - class_weight = balanced

# 3.2. Categorical Variable Encoding----------------------------------------------------------------------------

    # Separate out numeric and categorical variables
#numeric_cols = list(df2._get_numeric_data().columns)
#cat_cols = list(set(df2.columns)-set(numeric_cols)-{'Delayed_Arrival'})
    # NOTE: These will be used in the function which extracts feature and target arrays
#df2[cat_cols].head()

# 3.2.1. Time Data-------------------------------------------------
from datetime import datetime

# 1. Extract Date Info
flight_date = []
for date in df2['fl_date']:
    flight_date.append(datetime.strptime(date, "%Y-%m-%d"))

date = pd.Series(flight_date)
df2 = df2.merge(date.rename("flight_date"), left_index=True, right_index=True)
df2.head()

#3.2.2. Decompose date data into year, month, day

#year
flight_years = []
for date in df2['flight_date']:
    flight_years.append(date.year)

years = pd.Series(flight_years)
df2= df2.merge(years.rename("Year"), left_index=True, right_index=True)

# month
flight_month = []
for date in df2['flight_date']:
    flight_month.append(date.month)

month = pd.Series(flight_month)
df2= df2.merge(month.rename("Month"), left_index=True, right_index=True)

# day
flight_days = []
for date in df2['flight_date']:
    flight_days.append(date.day)

days = pd.Series(flight_days)
df2= df2.merge(days.rename("Day"), left_index=True, right_index=True)

#drop flight_date and fl_date
df2 = df2.drop(['fl_date'], axis=1)
df2 = df2.drop(['flight_date'], axis=1)
df2.head()

    # hour
df2['hr_dep'] = df2['crs_dep_time'] // 100 #Extract the hour of departure

    # create timr of day variable
time_of_day = []
for hour in df2['hr_dep']:
    if hour < 12:
        time_of_day.append("Morning")
    if hour >=12 and hour < 17:
        time_of_day.append("Afternoon")
    if hour >=17 and hour <= 24:
        time_of_day.append("Evening")

time_of_day_series = pd.Series(time_of_day)
df2 = df2.merge(time_of_day_series.rename("Time of Day"), left_index=True, right_index=True)
time_of_day_dummies = pd.get_dummies(df2['Time of Day'])
#merge season dummies to df
df2 = pd.concat([df2, time_of_day_dummies], axis=1)
df2 = df2.drop('Time of Day', axis=1)

# 3. Create Season Variable & Encode
seasons = []
for month in df2['Month']:
    if month <=2 or month > 11:
        seasons.append("Winter")
    if month >= 3 and month <= 5:
        seasons.append('Spring')
    if month >=6 and month <9:
        seasons.append('Summer')
    if month >= 9 and month <= 11:
        seasons.append('Fall')
season_series = pd.Series(seasons)
df2 = df2.merge(season_series.rename("Seasons"), left_index=True, right_index=True)
season_dummies = pd.get_dummies(df2['Seasons'])
#merge season dummies to df
df2 = pd.concat([df2, season_dummies], axis=1)
df2 = df2.drop('Seasons', axis=1)




# 3.2. Origin City
origin_city_dummies = pd.get_dummies(df2['origin_city_name'])
df2 = pd.concat([df2, origin_city_dummies], axis=1)
df2 = df2.drop('origin_city_name', axis=1)

# 3.3. Destination City
dest_city_dummies = pd.get_dummies(df2['dest_city_name'])
df2 = pd.concat([df2, dest_city_dummies], axis=1)
df2 = df2.drop('dest_city_name', axis=1)

# 3.4. Market Carrier
#mkt_carrier_dummies = pd.get_dummies(df2['mkt_carrier'])
#df2 = pd.concat([df2, mkt_carrier_dummies], axis=1)
df2 = df2.drop('mkt_carrier', axis=1)

# 3.5. Market Unique Carrier
#mkt_carrier_dummies = pd.get_dummies(df2['mkt_unique_carrier'])
#df2 = pd.concat([df2, mkt_carrier_dummies], axis=1)
df2 = df2.drop('mkt_unique_carrier', axis=1)

# 3.6 op_unique_carrier
op_unique_dummies = pd.get_dummies(df2['op_unique_carrier'])
df2 = pd.concat([df2, op_unique_dummies], axis=1)
df2 = df2.drop('op_unique_carrier', axis=1)

# 3.7. Origin
origin_dummies = pd.get_dummies(df2['origin'])
df2 = pd.concat([df2, origin_dummies ], axis=1)
df2 = df2.drop('origin', axis=1)

#3.8. Tail Number
#tail_dummies = pd.get_dummies(df2['tail_num'])
#df2 = pd.concat([df2, tail_dummies ], axis=1)
df2 = df2.drop('tail_num', axis=1)

#Drop Corrleated Variable
df3 = df2.drop('dep_delay', axis = 1)


# IV. MODELLING (BASE)================================================================================================

# 4.1. Split Training and Test Dataset Numeric Variable Transformation---------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix


numeric_cols = ['mkt_carrier_fl_num','op_carrier_fl_num', 'origin_airport_id', 'dest_airport_id', 'crs_dep_time', 'crs_arr_time', 'crs_elapsed_time', 'flights', 'distance']
cat_cols = list(set(df3.columns.values.tolist()) - set(df3[numeric_cols]) - {'Delayed_Arrival'})


scaler = StandardScaler()
scaler.fit(df3[numeric_cols]) 
X_scaled = scaler.transform(df3[numeric_cols])
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)
X_merge = pd.concat([X_scaled_df, df3[cat_cols]], axis=1)
X_merge2 = X_merge.drop([59972, 59973, 59974, 59974, 59975], axis=0)
X_merge3 = X_merge2.fillna(0)

# 4.2. Split Training and Test Dataset -------------------------------------------------------------------
y = df3['Delayed_Arrival']
X_train, X_test, y_train, y_test = train_test_split(X_merge3, y, test_size = 0.3, random_state=42, stratify=df3['Delayed_Arrival'])

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))  
# 4.3. Fit Logistic Regression Model-------------------------------------------

clf = LogisticRegression(penalty='none', class_weight='balanced', max_iter = 500) # log regression with no penaly term in the cost function
clf.fit(X_train, y_train)


# V. MODEL EVALUATION==========================================================================================================================================

#plot ROC curve
plot_roc_curve(clf, X_test, y_test)
#plot precision-recall curve
plot_precision_recall_curve(clf, X_test, y_test)

test_prob = clf.predict_proba(X_test)[:, 1]
test_pred = clf.predict(X_test)

def get_eval_metrics(y_test, test_prob, test_pred):
    print('Log loss = {:.5f}'.format(log_loss(y_test, test_prob)))
    print('AUC = {:.5f}'.format(roc_auc_score(y_test, test_prob)))
    print('Average Precision = {:.5f}'.format(average_precision_score(y_test, test_prob)))
    print('\nUsing 0.5 as threshold:')
    print('Accuracy = {:.5f}'.format(accuracy_score(y_test, test_pred)))
    print('Precision = {:.5f}'.format(precision_score(y_test, test_pred)))
    print('Recall = {:.5f}'.format(recall_score(y_test, test_pred)))
    print('F1 score = {:.5f}'.format(f1_score(y_test, test_pred)))

    print('\nClassification Report')
    print(classification_report(y_test, test_pred))
get_eval_metrics(y_test, test_prob, test_pred)

#Log loss = 0.67886
#AUC = 0.59775
#Average Precision = 0.22190

#Using 0.5 as threshold:
#Accuracy = 0.56497
#Precision = 0.22553
#Recall = 0.58863
#F1 score = 0.32611

#Classification Report
#              precision    recall  f1-score   support

#           0       0.86      0.56      0.68     24067
#           1       0.23      0.59      0.33      5241
#
#    accuracy                           0.56     29308
#   macro avg       0.54      0.57      0.50     29308
#weighted avg       0.75      0.56      0.62     29308


# Calculate Confuscion Matrix
print('Confusion Matrix')
plot_confusion_matrix(clf, X_test, y_test)


# VI. HYPERPARAMETER TUNING=========================================================================

# 1. GRID SEARCH w. K-FOLD CROSS VALIDATION
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(max_iter=500, solver='sag', class_weight='balanced')

#Setting the range for class weights
weights = np.linspace(0.0,0.99,200)

#Creating a dictionary grid for grid search
param_grid = {'class_weight': [{0:x, 1:1.0-x} for x in weights]}
param_grid2 = {'solver':['newtong-cg', 'lcbfgs', 'liblinear', 'sag', 'saga']}
param_grid3 = {'penalty': ['l2', 'none']}

#Fitting grid search to the train data with 5 folds
gridsearch = GridSearchCV(estimator= lr, 
                          param_grid= param_grid3,
                          cv=StratifiedKFold(), 
                          n_jobs=-1, 
                          scoring='f1', 
                          verbose=2).fit(X_train, y_train)

def grid_search_results(grid_search_obj):
    

#Ploting the score for different values of weight
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
weigh_data = pd.DataFrame({ 'score': gridsearch.cv_results_['mean_test_score'], 'weight': (1- weights)})
sns.lineplot(weigh_data['weight'], weigh_data['score'])
plt.xlabel('Weight for class 1')
plt.ylabel('F1 score')
plt.xticks([round(i/10,1) for i in range(0,11,1)])
plt.title('Scoring for different class weights', fontsize=24)

#-----

#importing and training the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='newton-cg', class_weight={0: 0.06467336683417085, 1: 0.9353266331658292})
lr.fit(x_train, y_train)

# Predicting on the test data
pred_test = lr.predict(x_test)

#Calculating and printing the f1 score 
f1_test = f1_score(y_test, pred_test)
print('The f1 score for the testing data:', f1_test)

#Ploting the confusion matrix
conf_matrix(y_test, pred_test)



#VII. INTERPRETING RESULTS===================================================================

import statsmodels.api as sm
log_reg = sm.Logit(y_train, X_train).fit()

from plot_classifier import plot_classifier
plt.figure(figsize=(6, 6))
plot_classifier(X_train, y_train, clf, ax=plt.gca())
plt.title("Logistic regression");





print("Model weights: %s"%(clf.coef_)) # these are weights
print("Model intercept: %s"%(clf.intercept_)) # this is the bias term
data = {'features': X_train.columns, 'coefficients':clf.coef_[0]}
pd.DataFrame(data)