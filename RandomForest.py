import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


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

    rfModel = RandomForestClassifier()
    rfParams = {
        'n_estimators' : [30],
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [4,7,10,20],
        'min_samples_leaf' : [40, 50, 75],
        'max_features' : ['sqrt', 'log2', 0.5],
        'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 4}]
    }
    rfModelCV = RandomizedSearchCV(rfModel, rfParams, n_iter=10, cv=4, verbose=1, n_jobs=-1, scoring='f1') # Try both f1 and accuracy
    rfModelCV.fit(xTrain, yTrain)
    print(rfModelCV.best_params_)
    print(rfModelCV.best_score_)
    pred = rfModelCV.predict(xTest)
    print(f1_score(yTest, pred))
    print(rfModelCV.score(xTest, yTest))
    plot_confusion_matrix(rfModelCV, xTest, yTest)
    plt.show()


if __name__ == "__main__":
    main()
