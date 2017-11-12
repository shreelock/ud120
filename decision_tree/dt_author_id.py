#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[0])

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
t_fit_i = time()
clf.fit(features_train, labels_train)
t_fit_f = time()


t_train_i = time()
labels_result = clf.predict(features_test)
t_train_f = time()

from sklearn.metrics import accuracy_score

#### your code goes here



acc = accuracy_score(labels_test, labels_result)
print "accuracy_score", acc
print "time result", t_train_f - t_train_i
print "time fit", t_fit_f - t_fit_i


#########################################################
### your code goes here ###


#########################################################


