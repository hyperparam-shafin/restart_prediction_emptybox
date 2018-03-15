
# import libraries required for data manipulation in python
import os
import pandas as pd
import numpy as np

# set working directory
os.chdir('C:\\pythondata\\restart_emptybox')

# check library versions
print("pandas " +str(pd.__version__))
print("numpy " +str(np.__version__))

# input balanced dataframes created with 100K samples for restarts and 100K samples for non restarts 
DF1 = pd.read_csv('restart1_K.csv', delimiter=',', low_memory=False,header=0)
print(DF1.shape) #shape of df with true restarts
DF2 = pd.read_csv('restart0_K.csv', delimiter=',', low_memory=False,header=0)
print(DF2.shape) #shape of df with non restarts
frames = [DF1,DF2] #list created for concatenation 
df_result = pd.concat(frames) #binded dataframe

# check resulting dataframe shape
print(df_result.shape)
df_result.head()
print(df_result.info())

# random sample of 20K
df_random = df_result.sample(20000, replace=False).copy()
print(df_random.shape)

# dropping dish_restart_flag as it is non significant
df_drop1 = df_random.drop(['dish_restart_flag'],axis=1).copy()
print(df_drop1.shape)

# typecasting variables
df_drop1['acct_no'] = df_drop1['acct_no'].astype('object')
df_drop1['commitment'] = df_drop1['commitment'].astype('object')

# identify the levels for each categorical variables 
feature_names = list(df_drop1.columns[1:23].values)
for column in feature_names:
    if df_drop1[column].dtypes=="object":
        print (column)
        print (df_drop1[column].value_counts(dropna=False))

# impute non viewer ship numerical factors and categorical or object type variable with mean and 'UNKNOWN", respectively
df_drop1['trds_mop_equal_2_or_gt_num'] = df_drop1.trds_mop_equal_2_or_gt_num.fillna(df_drop1.trds_mop_equal_2_or_gt_num.mean())
df_drop1['tot_amt_now_past_due_amt'] = df_drop1.tot_amt_now_past_due_amt.fillna(df_drop1.tot_amt_now_past_due_amt.mean())
df_drop1['curr_bal'] = df_drop1.curr_bal.fillna(df_drop1.curr_bal.mean())
df_drop1['core_programming'] = df_drop1.core_programming.fillna('UNKNOWN')
df_drop1['all_star'] = df_drop1.all_star.fillna('UNKNOWN')
df_drop1['payment_method'] = df_drop1.payment_method.fillna('UNKNOWN')
df_drop1['line_of_business'] = df_drop1.line_of_business.fillna('UNKNOWN')


# replace binary independent categorical variable with numeric data
print(df_drop1.core_international.value_counts())
df_drop1['core_international_ind'] = df_drop1.apply(lambda row: 1 if row['core_international']==" CORE INTERNATIONAL" else 0,axis=1)
print(df_drop1.core_international_ind.value_counts()) 

# drop the variable 
del df_drop1['core_international']

#check shape
df_drop1.shape

# create backup
df_scaled = df_drop1.copy()

# this function will help center my numerical variables x-mean(x)/std(x) with dof=0 
from sklearn.preprocessing import scale

# typecasting binary categorical variable
df['commitment'] = df['commitment'].astype('int64')

# scaled numerical variables and typecasting to float
df_scaled['count_of_latecharge_scale'] = scale(df_scaled['count_of_latecharge'].astype('float64'))
df_scaled['tenure_scale'] = scale(df_scaled['tenure'].astype('float64'))
df_scaled['credit_score_scale'] = scale(df_scaled['credit_score'].astype('float64'))
df_scaled['trds_mop_equal_2_or_gt_num_scale'] = scale(df_scaled['trds_mop_equal_2_or_gt_num'].astype('float64'))
df_scaled['tot_amt_now_past_due_amt_scale'] = scale(df_scaled['tot_amt_now_past_due_amt'].astype('float64'))
df_scaled['bill1_scale'] = scale(df_scaled['bill1'].astype('float64'))
df_scaled['bill2_scale'] = scale(df_scaled['bill2'].astype('float64'))
df_scaled['bill3_scale'] = scale(df_scaled['bill3'].astype('float64'))
df_scaled['bill4_scale'] = scale(df_scaled['bill4'].astype('float64'))
df_scaled['bill5_scale'] = scale(df_scaled['bill5'].astype('float64'))
df_scaled['curr_bal_scale'] = scale(df_scaled['curr_bal'].astype('float64'))
df_scaled['no_of_receivers_scale'] = scale(df_scaled['no_of_receivers'].astype('float64'))
df_scaled['total_devices_scale'] = scale(df_scaled['total_devices'].astype('float64'))
df_scaled['total_equip_charge_amt_scale'] = scale(df_scaled['total_equip_charge_amt'].astype('float64'))

# drop numerical variables that has been scaled successfully
df_scaled.drop(['count_of_latecharge','tenure','credit_score','trds_mop_equal_2_or_gt_num','tot_amt_now_past_due_amt',
'bill1','bill2','bill3','bill4','bill5','curr_bal','no_of_receivers','total_devices','total_equip_charge_amt'],axis=1)

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

# function to handle strings for dependent variable "is_restart"  
def dependent_col(row):
    if row['equip_status'] == 'RESTARTED':
        val = 1
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

# calling function to encode equip_status feature
df_scaled_cat['is_restart'] = df_scaled_cat.apply(dependent_col, axis=1)
df_scaled_cat.loc[:,['acct_no','equip_status','is_restart']].head()

# backup post scaling
df = df_scaled_cat.copy()

# drop categorical variables that has been encoded successfully
df = df.drop(['equip_status','core_programming','all_star','payment_method','line_of_business'],axis=1)

# set the first column i.e acct_no as index since it is both unique and should not be used for modeling
df_num = df.set_index('acct_no').copy()

# check shape needs to have exactly 21 columns with last column "is_restart"
df_num.shape

# check correlation matrix, high correlation wont affect model in a tree based algorithm 
cor = df_num.iloc[:,0:20].corr(method='pearson', min_periods=1)

# graphical packages installed
import seaborn as sns
import matplotlib.pyplot as plt

# generate a mask for the upper triangle
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# draw the heatmap with the mask and correct aspect ratio
sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

# converting features and dependent values to numpy arrays for making it model ready
array = df_num.values.copy()
X = array[:,0:20] # features
y = array[:,20]   # target

# import Logistic and recursive feature engineering to identify best columns 
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# to identify n best features when we have a wide dataset i.e with lots of independent features
# not required here because we have already done it in R project and the variables here are all
# contributing, at least to some degree
#model = LogisticRegression()
#rfe = RFE(model, 15)
#fit= rfe.fit(X,y)
#print("Num Features: " + str(fit.n_features_))
#print("Selected Features: " + str(fit.support_))
#print("Feature Ranking: " + str(fit.ranking_))

# import library to split data for modeling 
from sklearn.model_selection import train_test_split

# test set 30% and training set 70%
X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=21,test_size=0.3)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# import models to be tested
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import model_selection

# generate list of models  
models = []
models.append(('LR', LogisticRegression()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model
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

# untuned rf model
rf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
get_ipython().magic('timeit rf.fit(X_train,y_train)')
evaluate(rf,X_test,y_test)

# untuned knn model(not recommended for large datasets
knn=KNeighborsClassifier(3)
get_ipython().magic('timeit knn.fit(X_train,y_train)')
evaluate(knn,X_test,y_test)

# untuned adaboost model
ada = AdaBoostClassifier()
get_ipython().magic('timeit ada.fit(X_train,y_train)')
evaluate(ada,X_test,y_test)

# untuned gbm model
gbm = GradientBoostingClassifier()
get_ipython().magic('timeit gbm.fit(X_train,y_train)')
evaluate(gbm,X_test,y_test)

# untuned nn model
nn = MLPClassifier(alpha=1)
get_ipython().magic('timeit nn.fit(X_train,y_train)')
evaluate(nn,X_test,y_test)

#from pprint import pprint
#KNN = KNeighborsClassifier()
#KNN

# Look at parameters used by KNN
#print('Parameters currently in use:\n')
#pprint(KNN.get_params())

#from sklearn.model_selection import RandomizedSearchCV
# Number of neighbors
#n_neighbors = [int(x) for x in np.linspace(1, 20, num = 10)]
# algorithms to consider at every split
#algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
# Maximum number of leafs in tree
#leaf_size = [int(x) for x in np.linspace(10, 100, num = 10)]
# Minimum number of samples required to split a node
#p = [1, 2, 10]
# Minimum number of samples required at each leaf node
#metric = ['minkowski']
# Method of selecting samples for training each tree
#n_jobs = [-1]
#weights = ['uniform','distance']
# Create the random grid
#random_grid = {'n_neighbors': n_neighbors,
#              'algorithm': algorithm,
#               'leaf_size': leaf_size,
#               'n_jobs': n_jobs,
#               'p': p,
#               'weights':weights,
#               'metric': metric}
#pprint(random_grid)


#KNN = KNeighborsClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
#knn_random = RandomizedSearchCV(estimator = KNN, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
#knn_random.fit(X_train, y_train)

#knn_best = KNeighborsClassifier(algorithm = 'ball_tree', leaf_size = 90, metric = 'minkowski', n_jobs = -1,
#                                n_neighbors = 17, p = 2, weights = 'uniform')
#get_ipython().magic('timeit knn_best.fit(X_train, y_train)')
#knn_best_accuracy = evaluate(knn_best,X_test,y_test)

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

gbm_random.best_estimator_

gbm_best = GradientBoostingClassifier(criterion ='friedman_mse', learning_rate = 0.05, loss ='exponential', max_depth = 4,
                                      min_samples_leaf = 20, min_samples_split = 5, n_estimators = 500, random_state = 21)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gbm_best, X_train, y_train, cv=5)
scores                                              

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
get_ipython().magic('timeit gbm_best.fit(X_train, y_train)')
gbm_best_accuracy = evaluate(gbm_best,X_test,y_test)

# Plot feature importance
feature_names=df_num.columns[0:20]
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
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
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

# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
get_ipython().magic('timeit rf_random.fit(X_train, y_train)')

rf_random.best_params_
rf_random.best_estimator_
rf_random.best_params_
rf_random.best_score_

rf_best = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini', max_depth=80, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=10, min_weight_fraction_leaf=0.0,
			n_estimators=500, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
rf_best.fit(X_train,y_train)
rf_best_accuracy = evaluate(rf_best,X_test,y_test)
scores = cross_val_score(rf_best, X_train, y_train, cv=5)
scores

# Plot feature importance
feature_names=df_num.columns[0:20]
feature_importance = rf_best.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])
print(sorted_idx)
print(pos)
print(type(np.array(feature_names)))
print(type(feature_importance))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])
plt.show()
print((np.array(feature_names)))
print(feature_importance)

# save the model to disk
filename = 'finalized_model_rf_best200K.sav'
pickle.dump(rf_best, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open('finalized_model_rf_best200K.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

# import pickle
# save the model to disk
filename = 'finalized_gbm_best200K.sav'
pickle.dump(gbm_best, open(filename, 'wb'))


# In[419]:


# load the model from disk
loaded_model_gbm = pickle.load(open('finalized_gbm_best200K.sav', 'rb'))
result = loaded_model_gbm.score(X_test, y_test)
print(result)
loaded_model_rf = pickle.load(open('finalized_rf_best200K.sav', 'rb'))
result = loaded_model_rf.score(X_test, y_test)
print(result)
