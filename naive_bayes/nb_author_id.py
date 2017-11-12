#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
import numpy as numpy
from sklearn.naive_bayes import GaussianNB
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t_fit_i = time()
clf.fit(features_train, labels_train)
t_fit_f = time()


t_train_i = time()
labels_result = clf.predict(features_test)
t_train_f = time()

from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, labels_result)

print score
print "Training time :", round(t_fit_f-t_fit_i), "sec"
print "Prediction time :", round(t_train_f-t_train_i), "sec"


# Training time = 2.0 sec
# Testing time = 0.0 sec

#########################################################
### your code goes here ###


#########################################################


