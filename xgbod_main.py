# -*- coding: utf-8 -*-
"""Example of using XGBOD for outlier detection
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y
from scipy.io import loadmat

from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    # Define data file and read X and y
    # Generate some data if the source data is missing
    mat_file = 'cardio.mat'
    try:
        mat = loadmat(os.path.join('datasets/', mat_file))

    except TypeError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    except IOError:
        print('{data_file} does not exist. Use generated data'.format(
            data_file=mat_file))
        X, y = generate_data(train_only=True)  # load data
    else:
        X = mat['X']
        y = mat['y'].ravel()
        X, y = check_X_y(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=42)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    """
    train_data_batches, train_data_labels = [], []
    for batch in range(1, 6):
        data_dict = loadmat('datasets/cifar-10-batches-mat/data_batch_{}.mat'.format(batch))
        train_data_batches.append(data_dict['data'])
        train_data_labels.append(data_dict['labels'].flatten())
    train_set = {'images': np.concatenate(train_data_batches, axis=0),
                'labels': np.concatenate(train_data_labels, axis=0)}
    data_dict = loadmat('datasets/cifar-10-batches-mat/test_batch.mat')
    test_set = {'images': data_dict['data'],
                'labels': data_dict['labels'].flatten()}
    X_train = train_set['images']
    y_train = train_set['labels']
    X_test = test_set['images']
    y_test = test_set['labels']
    """
    clf_name = 'XGBOD'
    clf = XGBOD(random_state=42)
    clf.fit(X_train, y_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
    print('Test Data labels\n', y_test_pred)
    print(len(y_test_pred))
