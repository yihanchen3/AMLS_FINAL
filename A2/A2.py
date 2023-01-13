import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from Modules.result_process import *
import pandas as pd
from sklearn.datasets import load_iris


def SVC_origin(kernel='rbf', gamma='scale', c=1, verbose=1, tol=0.001, max_iter=8000):
    model = SVC(kernel=kernel, gamma=gamma, C=c, verbose=verbose, tol=tol, max_iter=max_iter)

    return model


def SVC_Grid(parameters, tol=0.1, cv=None):
    model = GridSearchCV(SVC(tol=tol), parameters, cv=cv, return_train_score=True)
    return model


def A2_SVC(x_train, y_train, x_test, y_test):
    model = SVC_origin()
    model.fit(x_train, y_train)
    train_acc = model.score(x_train, y_train)
    y_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred)

    plot_confusion_matrix('auto', y_pred, y_test)
    print("Test Accuracy: ", test_acc)
    return model, train_acc, test_acc


def A2_GridSVC(x_train, y_train, x_test, y_test):
    params = [
        {'kernel': ['linear'], 'C': [1, 10, 100]},
        {'kernel': ['poly'], 'C': [1], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [1, 0, 1, 0.01, 0.001]}
    ]
    model = SVC_Grid(params)
    model.fit(x_train, y_train)
    print("Best parameters set found on development set:\n", model.best_params_,
          "\n\nGrid scores on development set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    means_tr = model.cv_results_['mean_train_score']
    stds_tr = model.cv_results_['std_train_score']
    for mean_tr, std_tr, mean, std, params in zip(means_tr, stds_tr, means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) - %0.3f (+/-%0.03f) for %r" % (mean_tr, std_tr * 2, mean, std * 2, params))
    y_pred = model.predict(x_test)

    train_acc = model.score(x_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    plot_confusion_matrix('auto', y_pred, y_test)

    print("Accuracy:", train_acc, test_acc)
    return model, train_acc, test_acc
