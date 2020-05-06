# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:04:01 2020

@author: Ankit Pandey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

path = r"D:\AP\LinkedIn_Learning\Ex_Files_Machine_Learning_Algorithms\Ex_Files_Machine_Learning_Algorithms\Exercise Files"

titanic_data = pd.read_csv(path + "\\titanic.csv")

titanic_data.columns
titanic_data.head()

#Find the count of nulls across all columns
titanic_data.isnull().sum()

#Rwmoving the Missing values from Age
titanic_data['Age'].fillna(titanic_data["Age"].mean(), inplace = True)
titanic_data["Age"].head(10)

for i, col in enumerate(['Sex', 'Age', 'SibSp','Parch','Embarked']):
    plt.figure(i)
    sns.catplot(x=col, y ='Survived', data=titanic_data, kind='point', aspect = 2, )
#Since we are seeing a downward trend with Columns: SibSp and Parch,
#These both can be combined
titanic_data["ParentCnt"] = titanic_data["SibSp"] + titanic_data["Parch"]
#Dropping the two columns
titanic_data.drop(['SibSp','Parch'], axis = 1, inplace=True)

#Removing nulls from Cabin column
titanic_data.groupby(titanic_data['Cabin'].isnull())['Survived'].mean()
#Creating a model indicator
titanic_data["Cabin_ind"] = np.where(titanic_data["Cabin"].isnull(),0,1)


gender_num = {'male' : 0, 'female' : 1}

titanic_data["Sex"] = titanic_data["Sex"].map(gender_num)

titanic_data.columns

drp_cols = ["PassengerId","Cabin","Embarked","Name","Ticket"]
titanic_data.drop(drp_cols, inplace=True, axis=1)

titanic_data.head()

titanic_data.to_excel(path + "\cleaned_titanic_AP.xlsx", index=False)




# =============================================================================
# Dividing into Train - Validation - Test datasets
# =============================================================================

from sklearn.model_selection import train_test_split

features = titanic_data.drop("Survived", axis=1)
labels = titanic_data["Survived"]

#Train - 60%
#Test - 20%
#Validation - 20%
X_train, X_test, Y_train, Y_test = train_test_split(features, labels,test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,test_size=0.5, random_state=42)

for dat in [Y_train, Y_test, Y_val]:
    print(round(len(dat)/len(labels),2))

#writing to CSVs
X_train.to_excel(path + "\\train_features_AP.xlsx")
X_test.to_excel(path + "\\test_features_AP.xlsx")
X_val.to_excel(path + "\\val_features_AP.xlsx")

Y_train.to_excel(path + "\\train_labels_AP.xlsx")
Y_test.to_excel(path + "\\test_labels_AP.xlsx")
Y_val.to_excel(path + "\\val_labels_AP.xlsx")


# =============================================================================
# Application of Models after segregqation into Train - Validation - Test
# =============================================================================


# =============================================================================
# Logistic Regression 
# y = mx + b
#Here y is binary - Survived column  - 0 or 1
# =============================================================================

# =============================================================================
# Logistic - When to use
# 1. Binary target output/variable
# 2. When you are able to gauge the significance of predictor variables
# 3. Well behaved data - Not too many missing/outliers etc.
# 4. When you needa quick benchmark
# 5. DO NOT use when data volume is HIGH
# =============================================================================

#Hyperparameters of LR
from sklearn.linear_model import LogisticRegression

LogisticRegression()
'''
ALWAYS REMEMBER:
Best way to tune Hyperparameters is:
    NOT TUNE ALL hyperparameters
    Find the hyperparameters that have the maximum effect on the model
'''
dir(LogisticRegression())

#For LR - 
#    the hyperparameter C is the best option to be altered
#C = 1/ λ
# C is naturally called regularization parameter but,
#in reality λ is the regularization hyperparameter
#    Regularization is the   technique to reduce overfitting by discouraging 
#    overly complex models in some
# Inverse proportionality between λ and C (regularization parameter)
# Low values of C means HIGH REGULARIZATION leads to an UNDERFITTING model
# High values of C means LOW REGULARIZATION leads to OVERFITTING Model


import joblib
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Features
fs_train = X_train 

#Labels
lbls_train = Y_train



def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    
    for mean,stds,params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3),round(stds * 2,3),params))
        


lr = LogisticRegression()
parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
 }

cv = GridSearchCV(lr, parameters, cv = 5)
cv.fit(fs_train, lbls_train.values.ravel())

print_results(cv)
#
#This below is the best logistic regression model 
#that gives the best prediction
cv.best_estimator_

#It is important to save this so that it can be compared to other dofferent 
#models applied later
joblib.dump(cv.best_estimator_, path + "\\LR_Model_AP.pkl")



# =============================================================================
# Support Vector Machines 
# =============================================================================

#Using SVM:
#    usually when a classification problem has a binary target
#    Feature to row ratio is HIGH - More features(columns) less records(rows) - 
#    deals with complex relationships
#    data has many outliers
#    NOT a benchmark model - Takes time to train
    

from sklearn.svm import SVC

#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
#Out of the above Hyperparameters for SVM, the important one's are:
#    C and kernel
#"C" is the Penalty term hyperparameter: Determines how closely the model fits to the training set
#"kernel" is the trick to transform the plane so that it can be linearly seperable


#Applying k-Fold Analysis



tr_features = X_train
tr_labels = Y_train

#kernel trick - Controlling the trnsformation that makes the dataset linearly seperable

#using print_results() function

svc = SVC()
parameters = {
        'kernel': ['linear','rbf'],
        'C' : [0.1,1,10]
        }

cv = GridSearchCV(svc, parameters, cv = 5)
#fitting the model with cross validation = 5
cv.fit(tr_features, tr_labels.values.ravel())
#printing the results
print_results(cv)

# =============================================================================
# OUTPUT:
# BEST PARAMS: {'C': 0.1, 'kernel': 'linear'}
# 
# 0.796 (+/-0.116) for {'C': 0.1, 'kernel': 'linear'}
# 0.624 (+/-0.005) for {'C': 0.1, 'kernel': 'rbf'}
# 0.796 (+/-0.116) for {'C': 1, 'kernel': 'linear'}
# 0.667 (+/-0.081) for {'C': 1, 'kernel': 'rbf'}
# 0.796 (+/-0.116) for {'C': 10, 'kernel': 'linear'}
# 0.691 (+/-0.073) for {'C': 10, 'kernel': 'rbf'}
# 
# =============================================================================

cv.best_estimator_   
#SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

#Saving the SVM model
joblib.dump(cv.best_estimator_, path + "\\SVM_Model_AP.pkl")




# =============================================================================
'''# MULTI LAYER PERCEPTRON'''
# Classic feed-forward artificial neural network
#
#This means MLP is a series of connected nodes(i.e. functions) which are in the form of a directed 
#(means, never visited more than once) acyclic graph.
# =============================================================================

#When to use:
#    Useful for Classification and Regression
#    Performance oriented
#    Strong control over training process - Many Hyperparameters can be altered for results
#    No Transparency of process - More of Black box approach
#    NOT a quick benchmark model
#    NOT GOOD for limited data

from sklearn.neural_network import MLPRegressor, MLPClassifier

MLPRegressor()
#Hyperparameters
#MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(100,), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=None,
#       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#       verbose=False, warm_start=False)

MLPClassifier()
#Hyperparameters
#MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#       beta_2=0.999, early_stopping=False, epsilon=1e-08,
#       hidden_layer_sizes=(100,), learning_rate='constant',
#       learning_rate_init=0.001, max_iter=200, momentum=0.9,
#       nesterovs_momentum=True, power_t=0.5, random_state=None,
#       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
#       verbose=False, warm_start=False)

#WHOLE LOTTA HYPERPARAMETERS BHAGWAAN

#Important Hyperparameters:
#    ACTIVATION:
#        Activation function - this controls the type of non-linearity introduced in the model
#        Logistic Curve - Sigmoid - LogisticCurve in scikit-learn
#        TanH
#        ReLU - Rectified Linear Unit
#    HIDDEN_LAYER_SIZES:
#        Number of hidden layers and the number of nodes in each layer; 
#        this controls the complexity of the relationships that the model will capture
#    LEARNING_RATE:
#        This defines whether the model will find the OPTIMAL SOLUTION and how quickly will it do so
#        HIGH LR - Pushes the algo quicker to find the best solution - but we won't be sure that solution is optimal 
#        LOW LR - Will take time to find the optimal solution - but returns with best fit
#        
        
        

tr_features
tr_labels

mlp = MLPClassifier()

parameters = {
        'activation': ['relu','tanh', 'logistic'], #Signoid is same as logistic
        'hidden_layer_sizes': [(10,),(50,),(100,)], #below parameter means 10 nodes with 1 layer, 50 nodes with 1 layer and so on
        'learning_rate': ['constant', 'invscaling', 'adaptive']
        }


cv = GridSearchCV(mlp, parameters, cv = 5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)


print(cv.best_estimator_)

#Saving the MLP model
joblib.dump(cv.best_estimator_, path + "\\MLP_Model_AP.pkl")




# =============================================================================
'''# RANDOM FOREST'''
#Collection of independent decision trees to get a more accurate prediction
# =============================================================================

When to use RF algo:
    
from sklearn.ensemble import RandomForestClassifier
tr_features
tr_labels

rf = RandomForestClassifier()
#Hyperparameters for RF include:
#    (bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)

#But, for Hyperparameter Tuning, we will deal with only below:
#    n_estimators: 
#    max_depth: 

parameters = {
        'n_estimators': [5,50,100],
        'max_depth': [2,10,20,None]
        }

cv = GridSearchCV(rf,parameters, cv = 5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)
print(cv.best_estimator_)
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=10, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False)
