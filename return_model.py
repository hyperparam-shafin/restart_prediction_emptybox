
# import libraries required for data manipulation in python
import os
import pandas as pd
import numpy as np

# set working directory
os.chdir('C:\\pythondata\\return_emptybox')

# check library versions
print("pandas " +str(pd.__version__))
print("numpy " +str(np.__version__))

# input balanced dataframes created with 100K samples for restarts and 100K samples for non restarts 
DF1 = pd.read_csv('train_return1.csv', delimiter=',', low_memory=False,header=0)
print(DF1.shape) #shape of df with true restarts
DF2 = pd.read_csv('train_return0.csv', delimiter=',', low_memory=False,header=0)
print(DF2.shape) #shape of df with non restarts
frames = [DF1,DF2] #list created for concatenation 
df_result = pd.concat(frames) #binded dataframe

# check resulting dataframe shape
print(df_result.shape)
#df_result.head()

# random sample of 20K
#df_random = df_result.sample(20000, replace=False).copy()
#print(df_random.shape)


df_drop1 = df_result.copy()

# typecasting variables
df_drop1['acct_no'] = df_drop1['acct_no'].astype('object')
#df_drop1['restart_flag'] = df_drop1['restart_flag'].astype('object')
# identify the levels for each categorical variables 
feature_names = list(df_drop1.columns[2:17].values)
for column in feature_names:
    if df_drop1[column].dtypes=="object":
        print (column)
        print (df_drop1[column].value_counts(dropna=False))


# impute non viewer ship numerical factors and categorical or object type variable with mean and 'UNKNOWN", respectively
df_drop1['bnk_revlvng_trds_num'] = df_drop1.bnk_revlvng_trds_num.fillna(df_drop1.bnk_revlvng_trds_num.mean())
df_drop1['curr_bal'] = df_drop1.curr_bal.fillna(df_drop1.curr_bal.mean())
df_drop1['avg_bill'] = df_drop1.avg_bill.fillna(df_drop1.avg_bill.mean())
df_drop1['core_programming'] = df_drop1.core_programming.fillna('UNKNOWN')
df_drop1['all_star'] = df_drop1.all_star.fillna('UNKNOWN')
df_drop1['payment_method'] = df_drop1.payment_method.fillna('UNKNOWN')
df_drop1['line_of_business'] = df_drop1.line_of_business.fillna('UNKNOWN')
df_drop1['prime_post_office_name'] = df_drop1.prime_post_office_name.fillna('UNKNOWN')


# group by primary post office for replacement
df_drop1_gr = df_drop1.groupby('prime_post_office_name').mean()['avg_bill'].copy()
df_drop1_gr1 = pd.DataFrame(df_drop1_gr).copy()
df_drop1_gr1 = df_drop1_gr1.reset_index(inplace=False)
df_drop1_gr1 = df_drop1_gr1.rename(columns={'avg_bill': 'postal_bill'},inplace=False)
df_merge = pd.merge(left = df_drop1, right = df_drop1_gr1, on = 'prime_post_office_name')
df_merge.head()
df_merge.loc[:,['prime_post_office_name','postal_bill','avg_bill']].head(5)

# check mean bill 
mean_bill =np.mean(df_merge['postal_bill'])
mean_bill

# fix data type for arithmetic operation
df_merge['postal_bill'] = df_merge['postal_bill'].astype('float32')
df_merge['avg_bill'] = df_merge['avg_bill'].astype('float32')

# new column created
df_merge['postal_bill_new'] = df_merge.apply(lambda row: mean_bill if row['postal_bill']==row['avg_bill'] else row['postal_bill'],axis=1)

# check new value
df_merge.loc[:,['prime_post_office_name','postal_bill_new','postal_bill','avg_bill']].tail()

# drop pre-format variables
df_drop2 = df_merge.drop(['postal_bill','prime_post_office_name'],axis=1).copy()

# create backup
df_scaled = df_drop2.copy()

# this function will help center my numerical variables x-mean(x)/std(x) with dof=0 
from sklearn.preprocessing import scale

# scaled numerical variables and typecasting to float
df_scaled['count_of_late_charge_scale'] = scale(df_scaled['count_of_late_charge'].astype('float64'))
df_scaled['avg_bill_scale'] = scale(df_scaled['avg_bill'].astype('float64'))
df_scaled['bnk_revlvng_trds_num_scale'] = scale(df_scaled['bnk_revlvng_trds_num'].astype('float64'))
df_scaled['bill1_scale'] = scale(df_scaled['bill1'].astype('float64'))
df_scaled['bill2_scale'] = scale(df_scaled['bill2'].astype('float64'))
df_scaled['bill3_scale'] = scale(df_scaled['bill3'].astype('float64'))
df_scaled['bill4_scale'] = scale(df_scaled['bill4'].astype('float64'))
df_scaled['bill5_scale'] = scale(df_scaled['bill5'].astype('float64'))
df_scaled['curr_bal_scale'] = scale(df_scaled['curr_bal'].astype('float64'))
df_scaled['equip_charge_amt_scale'] = scale(df_scaled['equip_charge_amt'].astype('float64'))

df_scaled['postal_bill_new_scale'] = scale(df_scaled['postal_bill_new'].astype('float64'))

# drop variables which have been scaled
del df_scaled['count_of_late_charge']
del df_scaled['avg_bill']
del df_scaled['bnk_revlvng_trds_num']
del df_scaled['bill1']
del df_scaled['bill2']
del df_scaled['bill3']
del df_scaled['bill4']
del df_scaled['bill5']
del df_scaled['curr_bal']
del df_scaled['equip_charge_amt']
del df_scaled['postal_bill_new']

# backup for cleaned data ready for object to category encoding
df_scaled_cat = df_scaled.copy()

# function to handle strings for core_programming
def independent_col1(row):
    if row['core_programming'] == 'AT 200':
        val = 1
    elif row['core_programming'] == 'AT 120':
        val = 2
    elif row['core_programming'] == 'OTHER PROGRAMMING':
        val = 3
    elif row['core_programming'] == 'AT 250':
        val = 4
    elif row['core_programming'] == 'AT 120+':
        val = 5
    elif row['core_programming'] == 'Smart Pack':
        val = 6
    elif row['core_programming'] == 'Latino Dos':
        val = 7
    elif row['core_programming'] == 'UNKNOWN':
        val = 8
    elif row['core_programming'] == 'Latino Plus':
        val = 9
    elif row['core_programming'] == 'Americas Everything Pack':
        val = 10
    elif row['core_programming'] == 'Dish America':
        val = 11
    elif row['core_programming'] == 'Dish Mexico':
        val = 12
    elif row['core_programming'] == 'Latino Clasico':
        val = 13
    else:
        val = 0
    return val

# function to handle strings for all_star
def independent_col2(row):
    if row['all_star'] == '1_Star':
        val = 1
    elif row['all_star'] == '2_Star':
        val = 2
    elif row['all_star'] == '3_Star':
        val = 3
    elif row['all_star'] == '4_Star':
        val = 4
    elif row['all_star'] == '5_Star':
        val = 5
    else:
        val = 0
    return val


# function to handle strings for payment_method
def independent_col3(row):
    if row['payment_method'] == 'Standard Billing':
        val = 1
    elif row['payment_method'] == 'Recurring Credit Card':
        val = 2
    elif row['payment_method'] == 'EFT':
        val = 3
    else:
        val = 0
    return val

# function to handle strings for line_of_business
def independent_col4(row):
    if row['line_of_business'] == 'Video Only':
        val = 1
    elif row['line_of_business'] == 'Wildblue with Video':
        val = 2
    elif row['line_of_business'] == 'DSL with Video':
        val = 3
    elif row['line_of_business'] == 'NOT AVAILABLE':
        val = 4
    elif row['line_of_business'] == 'DSL and Telephone with Video':
        val = 5
    elif row['line_of_business'] == 'Wildblue Only':
        val = 6
    elif row['line_of_business'] == 'DSL Only':
        val = 7
    elif row['line_of_business'] == 'Telephone Only':
        val = 8
    elif row['line_of_business'] == 'Telephone with Video':
        val = 9
    elif row['line_of_business'] == 'DSL and Telephone only':
        val = 10
    else:
        val = 0
    return val


# calling function to encode core_programming feature
df_scaled_cat['core_programming_ind'] = df_scaled_cat.apply(independent_col1, axis=1)
df_scaled_cat.loc[:,['acct_no','core_programming','core_programming_ind']].head()

# calling function to encode all_star feature
df_scaled_cat['all_star_ind'] = df_scaled_cat.apply(independent_col2, axis=1)
df_scaled_cat.loc[:,['acct_no','all_star','all_star_ind']].head()

# calling function to encode payment_method feature
df_scaled_cat['payment_method_ind'] = df_scaled_cat.apply(independent_col3, axis=1)
df_scaled_cat.loc[:,['acct_no','payment_method','payment_method_ind']].head()

# calling function to encode line_of_business feature
df_scaled_cat['line_of_business_ind'] = df_scaled_cat.apply(independent_col4, axis=1)
df_scaled_cat.loc[:,['acct_no','line_of_business','line_of_business_ind']].head()


# drop categorical variables that has been encoded successfully
df_scaled_cat = df_scaled_cat.drop(['core_programming','all_star','payment_method','line_of_business'],axis=1)

# function to handle strings for dependent variable "is_return"  
def dependent_col(row):
    if row['equip_status'] == 'RETURNED':
        val = 1
    else:
        val = 0
    return val

# calling function to encode equip_status feature
df_scaled_cat['is_return'] = df_scaled_cat.apply(dependent_col, axis=1)
df_scaled_cat.loc[:,['acct_no','equip_status','is_return']].head()


del df_scaled_cat['equip_status']

# set the first column i.e acct_no as index since it is both unique and should not be used for modeling
df = df_scaled_cat.set_index('acct_no').copy()

# convert df to numpy array 
array = df.values.copy()
X = array[:,0:16] # features
y = array[:,16]   # target


from sklearn.model_selection import train_test_split

# test set 30% and training set 70%
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=21,test_size=0.3)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# import necessary packages
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection
import matplotlib.pyplot as plt

# generate list of models  
models = []
models.append(('LR', LogisticRegression()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('ADA', AdaBoostClassifier()))


# check model performance
results = []
names = []
seed = 7
scoring='accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = (name, cv_results.mean(), cv_results.std())
    print (msg)

# compare algorithms
fig = plt.figure()
fig.suptitle("algorithm comparision")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

from pprint import pprint
GBM = GradientBoostingClassifier()
GBM

# Look at parameters used by KNN
#print('Parameters currently in use:\n')
#pprint(KNN.get_params())
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(200, 500, num = 10)]
learning_rate = [0.1, 0.05, 0.02, 0.01]
max_features = ['sqrt','auto','log2','None','1','0.1']
loss = ['deviance', 'exponential']
max_depth = [4, 6, 8] 
criterion = ['friedman_mse']
min_samples_split = [2, 5, 10]
min_samples_leaf = [20,50,100,150]
random_state = [21]
# Create the random grid
random_grid_gbm = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'loss':loss,
               'max_depth': max_depth,
               'criterion': criterion,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'random_state': random_state}
pprint(random_grid_gbm)


GBM = GradientBoostingClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gbm_random = RandomizedSearchCV(estimator = GBM, param_distributions = random_grid_gbm, 
                                n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
gbm_random.fit(X_train, y_train)


gbm_best = gbm_random.best_estimator_

gbm_best.fit(X_train,y_train)

# function to evaluate model performance
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    report = classification_report(y_test, y_pred)
    print("accuracy :" +str(accuracy))
    print("matrix :")
    print(matrix)
    print ("tn, fp, fn, tp")
    print (tn, fp, fn, tp)
    print("report :")
    print(report)

gbm_best_accuracy = evaluate(gbm_best,X_test,y_test)

feature_names=df.columns[0:16]
feature_importance = gbm_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
print(sorted_idx)
print(pos)
print(type(np.array(feature_names)))
print(type(feature_importance))

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])

feature_names[sorted_idx]
plt.show()
print((np.array(feature_names)))
print(feature_importance)

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
rf_best = rf_random.best_estimator_
rf_best.fit(X_train,y_train)
rf_best_accuracy = evaluate(rf_best, X_test,y_test)


import matplotlib.pyplot as plt

feature_names=df.columns[0:16]
feature_importance = rf_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
print(sorted_idx)
print(pos)
print(type(np.array(feature_names)))
print(type(feature_importance))

plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])

feature_names[sorted_idx]
plt.show()
print((np.array(feature_names)))
print(feature_importance)

import pickle
# save the model to disk
filename = 'return_rf_best200K.sav'
pickle.dump(rf_best, open(filename, 'wb'))

# import pickle
# save the model to disk
filename = 'return_gbm_best200K.sav'
pickle.dump(gbm_best, open(filename, 'wb'))

# load the model from disk
loaded_model_gbm = pickle.load(open('return_gbm_best200K.sav', 'rb'))
result = loaded_model_gbm.score(X_test, y_test)
print(result)
loaded_model_rf = pickle.load(open('return_rf_best200K.sav', 'rb'))
result = loaded_model_rf.score(X_test, y_test)
print(result)

