# -*- coding: utf-8 -*-
"""Example of using XGBOD for outlier detection with PCA for dimensionality reduction"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.xgbod import XGBOD
from pyod.utils.data import evaluate_print

if __name__ == "__main__":
    # Load CIFAR-10 dataset
    train_data_batches, train_data_labels = [], []
    for batch in range(1, 6):  # CIFAR-10 typically has 5 training batches
        print(f"[+] Loading dataset batch: {batch}")
        data_dict = loadmat(f'datasets/cifar-10-batches-mat/data_batch_{batch}.mat')
        train_data_batches.append(data_dict['data'])
        train_data_labels.append(data_dict['labels'].flatten())

    train_set = {
        'images': np.concatenate(train_data_batches, axis=0),
        'labels': np.concatenate(train_data_labels, axis=0)
    }

    data_dict = loadmat('datasets/cifar-10-batches-mat/test_batch.mat')
    print("[+] Loaded the test dataset")
    test_set = {
        'images': data_dict['data'],
        'labels': data_dict['labels'].flatten()
    }

    X_train = train_set['images'] / 255.0  # Normalize to [0, 1]
    y_train = train_set['labels']
    X_test = test_set['images'] / 255.0
    y_test = test_set['labels']

    # Standardize features (mean=0, variance=1) for PCA
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA for dimensionality reduction
    n_components = 100  # You can adjust this or use n_components=0.95 for variance-based selection
    print(f"[+] Applying PCA to reduce dimensions to {n_components} components")
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"[+] PCA applied: New training shape: {X_train.shape}, New test shape: {X_test.shape}")

    # Initialize and train XGBOD with GPU support
    clf_name = 'XGBOD'
    clf = XGBOD(
        random_state=42,
        verbosity=1,
        tree_method='hist',
        device='cuda',
        n_jobs=-1,  # Use all available cores
        n_estimators=20  # Reduce estimators for faster training
    )

    print("[+] Training XGBOD model...")
    clf.fit(X_train, y_train)
    print("[+] Model training completed.")

    # Predictions and outlier scores
    y_train_pred = clf.labels_
    print(f"y train pred: {y_train_pred}")
    y_train_scores = clf.decision_scores_
    print(f"y train scores: {y_train_scores}")

    print("predicting for test")
    y_test_pred = clf.predict(X_test)
    print(f"y_test_pred: {y_test_pred}")
    y_test_scores = clf.decision_function(X_test)
    print(f"test_score {y_test_scores}")
    print("Exiting.....")

    # Evaluation
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)

    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)
    print(f"Test Data Predicted Labels:\n{y_test_pred}")
    print(f"Total Predictions: {len(y_test_pred)}")
