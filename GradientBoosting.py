import argparse
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
    parser.add_argument("--seed", default=336,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain).ravel()
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest).ravel()

    scaler = StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    np.random.seed(args.seed)

    gbModel = GradientBoostingClassifier(n_estimators=5)
    gbParams = {
        'loss': ['deviance'],
        'learning_rate': [1.0],
        'criterion': ['mse'],
        'max_depth': [250,500],
        'min_samples_leaf': [1],
        'max_features': [0.5]
    }
    gbModelCV = GridSearchCV(gbModel, gbParams, cv=3, verbose=1, n_jobs=-1,
                                   scoring='f1')  # Try both f1 and accuracy
    gbModelCV.fit(xTrain, yTrain)
    print(gbModelCV.best_params_)
    print(gbModelCV.best_score_)
    pred = gbModelCV.predict(xTest)
    print(f1_score(yTest, pred))
    print(gbModelCV.score(xTest, yTest))
    plot_confusion_matrix(gbModelCV, xTest, yTest)
    plt.show()


if __name__ == "__main__":
    main()
