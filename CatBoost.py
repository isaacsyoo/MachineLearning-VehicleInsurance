import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


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
    parser.add_argument("--seed", default=337,
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

    cbModel = CatBoostClassifier()
    cbParams = {
        'max_depth': [3,4,5],
        'iterations': [10, 25, 50, 100],
        'learning_rate': [0.25, 0.33, 0.5, 0.75],
        'scale_pos_weight': [3, 4, 5]
    }
    cbModelCV = RandomizedSearchCV(cbModel, cbParams, n_iter=100, cv=4, verbose=1, n_jobs=-1, scoring='roc_auc') # Try both f1 and accuracy
    cbModelCV.fit(xTrain, yTrain)
    print(cbModelCV.best_params_)
    print(cbModelCV.best_score_)
    pred = cbModelCV.predict(xTest)
    print(f1_score(yTest, pred))
    print(cbModelCV.score(xTest, yTest))
    plot_confusion_matrix(cbModelCV, xTest, yTest)
    plt.show()


if __name__ == "__main__":
    main()
