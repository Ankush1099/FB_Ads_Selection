# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:34:19 2020

@author: Ankush
"""


# Step1: Import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step2: Import the dataset
training_set = pd.read_csv('Facebook_Ads_2.csv', encoding = 'ISO-8859-1')
training_set.head(5)
training_set.tail(5)

#Step3: Visualize Dataset
click = training_set[training_set['Clicked']==1]
no_click = training_set[training_set['Clicked']==0]
print('Total = ', len(training_set))
print('Total number of passengers who clicked = ', len(click))
print('Total number of passengers who didnot clicked= ', len(no_click))

sns.scatterplot(training_set["Time Spent on Site"], training_set["Salary"], hue = training_set['Clicked'])

sns.boxplot(x='Clicked', y='Salary', data=training_set)
sns.boxplot(x='Clicked', y='Time Spent on Site', data=training_set)

training_set['Salary'].hist(bins = 40)
training_set['Time Spent on Site'].hist(bins = 20)

#Step4: Data Cleaning
training_set.drop(['Names','emails','Country'], axis = 1, inplace = True)
training_set

X = training_set.drop(['Clicked'], axis = 1).values
X
y = training_set['Clicked'].values
y

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Step5: Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Step6: Model Testing
#Training data prediction
y_predict_train = classifier.predict(X_train)
y_predict_train
y_train

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True, fmt = 'd')

#Testing data prediction
y_predict_test = classifier.predict(X_test)
y_predict_test
y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True, fmt = 'd')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_test))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Testing set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
