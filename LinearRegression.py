import argparse
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    parser.add_argument("--seed", default=333,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain).ravel()
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest).ravel()

    np.random.seed(args.seed)

    scaler = MinMaxScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    lrModel = LogisticRegression(solver='saga', max_iter=100)
    lrParams = {
        'penalty': ['l1', 'l2', 'none'],
        'C': uniform(loc=0, scale=4)
    }
    lrModelCV = RandomizedSearchCV(lrModel, lrParams, n_iter=10, cv=3, verbose=1, n_jobs=-1, scoring='f1')
    lrModelCV.fit(xTrain, yTrain)
    print(lrModelCV.best_params_)
    print(lrModelCV.best_score_)
    print(lrModelCV.score(xTrain, yTrain))



if __name__ == "__main__":
    main()
