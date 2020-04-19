# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:28:52 2020

@author: Katayoon Mehr
"""

### Multi Class y - Classification ###
''' Logistic Regression, KNN, adaboost, Random Forest, SVM '''

import numpy as np
import pandas as pd
import os
os.chdir('C:\\Users\\Rayan\\Desktop\\Kati\\Machine Learning\\Project\\Class')
#os.chdir('E:\\Machine Learning\\Project')
os.getcwd()

Salary=pd.read_csv('Adult_Salary_KNN.csv', delimiter=',')

Salary.head()
Salary.describe()
Salary.columns
Salary.info()
Salary.isnull().any()
Salary.isnull().sum()

# Handling Missing values
Salary['native_country'].fillna(value='Unknown', axis=0, inplace= True)
Salary['origin'].fillna(value='Unknown', axis=0, inplace= True)
Salary['occupation'].fillna(value='Unknown', axis=0, inplace= True)
Salary['workclass'].fillna(value='Unknown', axis=0, inplace= True)

# Encoding
Salary['marital_status_C']=Salary.marital_status.map({'Divorced':1,
                                                      'Married':2, 
                                                      'Never-married':3,
                                                      'Separated':4,
                                                      'Widowed':5})



Salary['workclass_C']=Salary.workclass.map({'Federal-gov':1,
                                            'Local-gov':2,
                                            'State-gov':3,
                                            'Private':4,
                                            'Self-emp-inc':5,
                                            'Self-emp-not-inc':6,
                                            'Without-pay':7,
                                            'Never-worked':8,
                                            'Unknown':9})

Salary['race_C']=Salary.race.map({'White':1,
                                  'Black':2, 
                                  'Asian':3,
                                  'Indian':4,
                                  'Other':5})

Salary['gender_C']=Salary.gender.map({'Male':1,
                                      'Female':2})

Salary['origin_C']=Salary.origin.map({'North America':1,
                                      'South America':2, 
                                      'Asia':3,
                                      'Europe':4,
                                      'Africa':5,
                                      'Unknown':6})



ax1 = sns.boxplot(x=Salary.inc_c, y=Salary.age)
ax1.set(xlabel='Income 1:<25k, 2:25to50, 3:50-100, 4:100-150, 5:>150',ylabel='Age', title='Distribution of Age and income')
ax2 = sns.boxplot(x=Salary.origin, y=Salary.age)
ax3 = sns.boxplot(x=Salary.origin, y=Salary.inc_c)
ax3.set(ylabel='Income', title='Distribution of Income and Origin')


y = Salary['inc_c']
X = Salary[['age', 'education_y', 'marital_status_C', 'workclass_C', 'race_C', 'gender_C', 'hpw', 'origin_C']]

import seaborn as sns
import matplotlib.pyplot as plt

X_Corr = X.corr()
sns.heatmap(X.corr(), cmap="YlGnBu")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train_S=sc.fit_transform(X_train)
X_test_S=sc.transform(X_test)



''' Logistic Regression '''

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, recall_score, \
                         precision_score, confusion_matrix, classification_report
                         
model = LogisticRegression()
model.fit(X_train_S,y_train)

model.intercept_
model.coef_

y_pred_lg = model.predict(X_test_S)

R2_train=model.score(X_train_S, y_train)
R2_test=model.score(X_test_S, y_test)


mnlogit = sm.MNLogit(y_train, X_train_S)
model = mnlogit.fit()
print(model.summary())

cm_lg = confusion_matrix(y_test, y_pred_lg)
print(classification_report(y_test, y_pred_lg))



''' KNN '''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, \
                         precision_score, confusion_matrix, classification_report
                         
model = KNeighborsClassifier(n_neighbors=7, metric = 'minkowski', weights = 'uniform', p=2)

model.fit(X_train_S, y_train)
y_pred_knn = model.predict(X_test_S)

cm_knn = confusion_matrix(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))

score=[]
for i in range (1, 15):
    classifier = KNeighborsClassifier(n_neighbors = i, weights='uniform', p = 2)
    classifier.fit(X_train_S, y_train)
    sc=classifier.score(X_test_S, y_test)
    score.append(sc)

import matplotlib.pyplot as plt
plt.plot(range(1,15), score)
plt.xlabel("no of K")
plt.ylabel("Accuracy on the test dataset")
plt.show()



''' Random Forest '''

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

rf = RandomForestClassifier(n_estimators=1000, random_state=0)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

y_pred_rf = rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)  
print(classification_report(y_test, y_pred_rf))


'''
errors = abs(y_pred_rf - y_test)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
Precision = 100 - np.mean(mape)
print('Precision:', round(accuracy, 2), '%.')

from sklearn.tree import export_graphviz
# Pull out one tree from the forest
tree = rf.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', rounded = True, precision = 1)
(graph, ) = pyd.graph_from_dot_file('tree.dot')
'''


scores=[]
for i in [10, 100, 1000, 2500]:
    rf = RandomForestClassifier(n_estimators=i, random_state=10)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))   
    
import matplotlib.pyplot as plt
plt.plot([10, 100, 1000, 2500], scores)
plt.xlabel("no of modelss")
plt.ylabel("Accuracy on the test dataset")
plt.show()


''' Adaboost ''' 

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=100, random_state=10)
ada.fit(X_train_S, y_train)
ada.score(X_test_S, y_test)

y_pred_ada = ada.predict(X_test_S)
cm_ada = confusion_matrix(y_test, y_pred_ada)  
print(classification_report(y_test, y_pred_ada))

scores=[]
for i in [10, 50, 100, 250]:
    ada = AdaBoostClassifier(n_estimators=i, random_state=10)
    ada.fit(X_train_S, y_train)
    scores.append(ada.score(X_test_S, y_test))  
    
import matplotlib.pyplot as plt
plt.plot([10, 50, 100, 250], scores)
plt.xlabel("no of models")
plt.ylabel("Accuracy on the test dataset")
plt.show()




''' SVM '''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, \
                         precision_score, confusion_matrix, classification_report
                         
svm = SVC(kernel = 'rbf', gamma = 10 , C=10)
svm.fit(X_train_S, y_train)
y_pred_svm = svm.predict(X_test_S)

cm_cvm = confusion_matrix(y_test, y_pred_svm)
print(classification_report(y_test, y_pred_svm))


from sklearn.model_selection import GridSearchCV
param_dict = {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma' : [0.1,1,10],
                'C': [0.1,1,10]                
            }
grid = GridSearchCV(SVC(), param_dict, cv=4)
grid.fit(X_train_S, y_train)
grid.best_params_
grid.best_score_


###################################################################################3