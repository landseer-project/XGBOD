# -*- coding: utf-8 -*-
"""Example of using XGBOD for outlier detection with PCA for dimensionality reduction"""

from __future__ import division, print_function
import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pyod.models.xgbod import XGBOD

if __name__ == "__main__":
    train_data_batches, train_data_labels = [], []
    for batch in range(1, 6):  # CIFAR-10 has 5 training batches
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

    X_train = train_set['images'] / 255.0  
    y_train = train_set['labels']
    X_test = test_set['images'] / 255.0
    y_test = test_set['labels']

    y_train_binary = (y_train != 0).astype(int)
    y_test_binary = (y_test != 0).astype(int)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_components = 100
    print(f"[+] Applying PCA with {n_components} components")
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"[+] PCA applied: Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    clf = XGBOD(
        random_state=42,
        verbosity=1,
        tree_method='hist',
        device='cuda',
        n_jobs=-1,
        n_estimators=20
    )

    print("[+] Training XGBOD model...")
    clf.fit(X_train, y_train_binary)
    print("[+] Model training completed.")

    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    print("\n=== Evaluation Metrics ===")

    train_roc = roc_auc_score(y_train_binary, y_train_scores)
    train_precision = precision_score(y_train_binary, y_train_pred)
    train_recall = recall_score(y_train_binary, y_train_pred)
    train_f1 = f1_score(y_train_binary, y_train_pred)

    print("\nOn Training Data:")
    print(f"ROC AUC: {train_roc:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1 Score: {train_f1:.4f}")

    test_roc = roc_auc_score(y_test_binary, y_test_scores)
    test_precision = precision_score(y_test_binary, y_test_pred)
    test_recall = recall_score(y_test_binary, y_test_pred)
    test_f1 = f1_score(y_test_binary, y_test_pred)

    print("\nOn Test Data:")
    print(f"ROC AUC: {test_roc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    print("\n=== Prediction Summary ===")
    print(f"Test Data Predicted Labels: {y_test_pred}")
    print(f"Total Predictions: {len(y_test_pred)}")
    np.savetxt("xgbod_out.txt", y_test_pred, fmt="%d")
    print("Predicted labels saved to xgbod_out.txt")
