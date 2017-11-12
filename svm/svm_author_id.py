#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC

i=6000 # Found last time by looping
clf = SVC(C=i, kernel='rbf')

t_fit_i = time()
# clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
clf.fit(features_train, labels_train)
t_fit_f = time()


t_train_i = time()
labels_result = clf.predict(features_test)
t_train_f = time()

from sklearn.metrics import accuracy_score
score = accuracy_score(labels_test, labels_result)

print score
values, counts = np.unique(labels_result, return_counts=True)
print values, counts
print "Training time :", round(t_fit_f-t_fit_i), "sec"
print "Prediction time :", round(t_train_f-t_train_i), "sec"

# For the While dataset, with "linear" kernel
# Training time = 244.0 sec
# Testing time = 26.0 sec

# with Rbf kernel :
# We got highest accuracy = 0.90 at C=6000

# With rbf and c=6000, acc=0.99


