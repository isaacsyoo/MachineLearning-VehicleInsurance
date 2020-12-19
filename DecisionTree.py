import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import uniform
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain).ravel()
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest).ravel()

    np.random.seed(args.seed)

    scaler = StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    dtModel = DecisionTreeClassifier()
    dtParams = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [4,7,10,20,50,200],
        'min_samples_leaf' : [1,4,10,20,40],
        'class_weight': ['balanced', {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}, {0:1, 1:6}]
    }
    dtModelCV = RandomizedSearchCV(dtModel, dtParams, n_iter=1, cv=4, verbose=1, n_jobs=-1, scoring='f1') # Try both f1 and accuracy
    dtModelCV.fit(xTrain, yTrain)
    print(dtModelCV.best_params_)
    print(dtModelCV.best_score_)
    pred = dtModelCV.predict(xTest)
    print(f1_score(yTest, pred))
    print(dtModelCV.score(xTest, yTest))
    plot_confusion_matrix(dtModelCV, xTest, yTest)
    plt.show()

    """
    dtModel = DecisionTreeClassifier(min_samples_leaf=75, max_depth=10, criterion='gini', class_weight={0: 1, 1: 3})
    dtModel.fit(xTrain, yTrain)
    plot_confusion_matrix(dtModel, xTest, yTest)
    plt.show()

    dtModel = RandomForestClassifier(n_estimators=10, min_samples_leaf=50, max_depth=10, criterion='entropy',
                                 max_features=0.5, class_weight={0: 1, 1: 3})
    dtModel.fit(xTrain, yTrain)
    plot_confusion_matrix(dtModel, xTest, yTest)
    plt.show()

    dtModel = XGBClassifier(scale_pos_weight=3, n_estimators=50, max_depth=4, reg_lambda=1, gamma=0.1, eta=0.25, alpha=0)
    dtModel.fit(xTrain, yTrain)
    plot_confusion_matrix(dtModel, xTest, yTest)
    plt.show()

    dtModel = CatBoostClassifier(scale_pos_weight=3, max_depth=4, learning_rate=0.25, iterations=100)
    dtModel.fit(xTrain, yTrain)
    plot_confusion_matrix(dtModel, xTest, yTest)
    plt.show()
    """


if __name__ == "__main__":
    main()
