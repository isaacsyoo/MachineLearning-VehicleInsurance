import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, auc, f1_score, precision_recall_curve, \
    plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


# def roc_auc(yTest, yScoreRF, yScoreGB, yScoreLR, yScoreCB, yScoreXGB, yScoreLGBM):
def roc_auc(yTest, yScoreDT, yScoreRF, yScoreGB, yScoreXGB, yScoreCB):
    # roc_auc_score
    aucDT = roc_auc_score(yTest, yScoreDT)  # roc_auc_score for decision tree
    aucRF = roc_auc_score(yTest, yScoreRF)  # roc_auc_score for random forest
    aucGB = roc_auc_score(yTest, yScoreGB)  # roc_auc_score for gradient boosting
    # aucLR = roc_auc_score(yTest, yScoreLR) #roc_auc_score for logistic regression
    aucCB = roc_auc_score(yTest, yScoreCB) #roc_auc_score for CatBoost
    aucXGB = roc_auc_score(yTest, yScoreXGB) #roc_auc_score for XGBoost
    # aucLGBM = roc_auc_score(yTest, yScoreLGBM) #roc_auc_score for LGBM

    # print AUC scores
    print('AUC for Decision Tree: ' + str(aucDT))  # print AUC score for decision tree
    print('AUC for Random Forest: ' + str(aucRF))  # print AUC score for random forest
    print('AUC for Gradient Boosting: ' + str(aucGB))  # print AUC score for gradient boosting
    # print('AUC for Logistic Regression: ' + str(aucLR)) #print AUC score for logistic regression
    print('AUC for CatBoost: ' + str(aucCB)) #print AUC score for CatBoost
    print('AUC for XGBoost: ' + str(aucXGB)) #print AUC score for XGBoost
    # print('AUC for LightGBM: ' + str(aucLGBM)) #print AUC score for LightGBM

    # roc_curve
    fprDT, tprDT, thresholdDT = roc_curve(yTest, yScoreDT)  # roc_curve for decision tree
    fprRF, tprRF, thresholdRF = roc_curve(yTest, yScoreRF)  # roc_curve for random forest
    fprGB, tprGB, thresholdGB = roc_curve(yTest, yScoreGB)  # roc_curve for gradient boosting
    # fprLR, tprLR, thresholdLR = roc_curve(yTest, yScoreLR) #roc_curve for logistic regression
    fprCB, tprCB, thresholdCB = roc_curve(yTest, yScoreCB) #roc_curve for CatBoost
    fprXGB, tprXGB, thresholdXGB = roc_curve(yTest, yScoreXGB) #roc_curve for XGBoost
    # fprLGBM, tprLGBM, thresholdLGBM = roc_curve(yTest, yScoreLGBM) #roc_curve for LightGBM

    # plot ROC curves
    plt.plot([0, 1], ls="--")  # diagonal dashed lines
    plt.plot(fprDT, tprDT, label='Decision Tree')  # plot roc curve for decision tree
    plt.plot(fprRF, tprRF, label='Random Forest')  # plot roc curve for random forest
    plt.plot(fprGB, tprGB, label='Gradient Boosting')  # plot roc curve for gradient boosting
    # plt.plot(fprLR, tprLR, label = 'Logistic Regression') #plot roc curve for logistic regression
    plt.plot(fprCB, tprCB, label = 'CatBoost') #plot roc curve for CatBoost
    plt.plot(fprXGB, tprXGB, label = 'XGBoost') #plot roc curve for XGBoost
    # plt.plot(fprLGBM, tprLGBM, label = 'LightGBM') #plot roc curve for LightGBM
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


# def precision_recall(yTest, yScoreRF, yScoreGB, yScoreLR, yScoreCB, yScoreXGB, yScoreLGBM, yHatRF, yHatGB, yHatLR, yHatCB, yHatXGB, yHatLGBM):
def precision_recall(yTest, yScoreDT, yScoreRF, yScoreGB, yScoreXGB, yScoreCB, yHatDT, yHatRF, yHatGB, yHatXGB, yHatCB):
    # precision & recall
    precisionDT, recallDT, _ = precision_recall_curve(yTest, yScoreDT)  # precision_recall_curve for decision tree
    precisionRF, recallRF, _ = precision_recall_curve(yTest, yScoreRF)  # precision_recall_curve for random forest
    precisionGB, recallGB, _ = precision_recall_curve(yTest, yScoreGB)  # precision_recall_curve for gradient boosting
    # precisionLR, recallLR, _ = precision_recall_curve(yTest, yScoreLR) #precision_recall_curve for logistic regression
    precisionCB, recallCB, _ = precision_recall_curve(yTest, yScoreCB) #precision_recall_curve for CatBoost
    precisionXGB, recallXGB, _ = precision_recall_curve(yTest, yScoreXGB) #precision_recall_curve for XGBoost
    # precisionLGBM, recallLGBM, _ = precision_recall_curve(yTest, yScoreLGBM) #precision_recall_curve for LightGBM

    # f1 scores
    f1DT = f1_score(yTest, yHatDT)  # f1 score for decision tree
    f1RF = f1_score(yTest, yHatRF)  # f1 score for random forest
    f1GB = f1_score(yTest, yHatGB)  # f1 score for gradient boosting
    # f1LR = f1_score(yTest, yHatLR) #f1 score for logistic regression
    f1CB = f1_score(yTest, yHatCB) #f1 score for CatBoost
    f1XGB = f1_score(yTest, yHatXGB) #f1 score for XGBoost
    # f1LGBM = f1_score(yTest, yHatLGBM) #f1 score for LightGBM

    # print f1 scores
    print('Decision Tree: f1=%.3f' % (f1DT))  # f1 score for decision tree
    print('Random Forest: f1=%.3f' % (f1RF))  # f1 score for random forest
    print('Gradient Boosting: f1=%.3f' % (f1GB))  # f1 score for gradient boosting
    # print('Logistic Regression: f1=%.3f' % (f1LR)) #f1 score for logistic regression
    print('CatBoost: f1=%.3f' % (f1CB)) #f1 score for CatBoost
    print('XGBoost: f1=%.3f' % (f1XGB)) #f1 score for XGBoost
    # print('LightGBM: f1=%.3f' % (f1LGBM)) #f1 score for LightGBM

    # AUPRC scores
    f1DT = average_precision_score(yTest, yHatDT)  # f1 score for decision tree
    f1RF = average_precision_score(yTest, yHatRF)  # f1 score for random forest
    f1GB = average_precision_score(yTest, yHatGB)  # f1 score for gradient boosting
    # f1LR = f1_score(yTest, yHatLR) #f1 score for logistic regression
    f1CB = average_precision_score(yTest, yHatCB) #f1 score for CatBoost
    f1XGB = average_precision_score(yTest, yHatXGB) #f1 score for XGBoost
    # f1LGBM = f1_score(yTest, yHatLGBM) #f1 score for LightGBM

    # print f1 scores
    print('Decision Tree: AUPRC=%.3f' % (f1DT))  # f1 score for decision tree
    print('Random Forest: AUPRC=%.3f' % (f1RF))  # f1 score for random forest
    print('Gradient Boosting: AUPRC=%.3f' % (f1GB))  # f1 score for gradient boosting
    # print('Logistic Regression: f1=%.3f' % (f1LR)) #f1 score for logistic regression
    print('CatBoost: AUPRC=%.3f' % (f1CB)) #f1 score for CatBoost
    print('XGBoost: AUPRC=%.3f' % (f1XGB)) #f1 score for XGBoost
    # print('LightGBM: f1=%.3f' % (f1LGBM)) #f1 score for LightGBM

    # plot precision-recall curves
    plt.plot(recallDT, precisionDT, label='Decision Tree')  # plot precision-recall curve for decision tree
    plt.plot(recallRF, precisionRF, label='Random Forest')  # plot precision-recall curve for random forest
    plt.plot(recallGB, precisionGB, label='Gradient Boosting')  # plot precision-recall curve for gradient boosting
    # plt.plot(recallLR, precisionLR, label = 'Logistic Regression') #plot precision-recall curve for logistic regression
    plt.plot(recallCB, precisionCB, label = 'CatBoost') #plot precision-recall curve for CatBoost
    plt.plot(recallXGB, precisionXGB, label = 'XGBoost') #plot precision-recall curve for XGBoost
    # plt.plot(recallLGBM, precisionLGBM, label = 'LghtGBM') #plot precision-recall curve for LightGBM
    plt.title('Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


def main():
    # load CSV files
    xTrain = pd.read_csv("xTrain.csv")  # load xTrain
    xTest = pd.read_csv("xTest.csv")  # load xTest
    yTrain = pd.read_csv("yTrain.csv")  # load yTrain
    yTest = pd.read_csv("yTest.csv")  # load yTest

    # convert to numpy
    xTrain = xTrain.to_numpy()  # convert to numpy
    xTest = xTest.to_numpy()  # convert to numpy
    yTrain = yTrain.to_numpy().ravel()  # convert to numpy
    yTest = yTest.to_numpy().ravel()  # convert to numpy

    scaler = StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    yHatDT, yScoreDT = dt(xTrain, xTest, yTrain)
    yHatRF, yScoreRF = rf(xTrain, xTest, yTrain)
    yHatGB, yScoreGB = gb(xTrain, xTest, yTrain)
    yHatXGB, yScoreXGB = xgb(xTrain, xTest, yTrain)
    yHatCB, yScoreCB = cb(xTrain, xTest, yTrain)

    # roc_auc(yTest, yScoreRF, yScoreGB, yScoreLR, yScoreCB, yScoreXGB, yScoreLGBM)
    roc_auc(yTest, yScoreDT, yScoreRF, yScoreGB, yScoreXGB, yScoreCB)

    # precision_recall(yTest, yScoreRF, yScoreGB, yScoreLR, yScoreCB, yScoreXGB, yScoreLGBM, yHatRF, yHatGB, yHatLR, yHatCB, yHatXGB, yHatLGBM)
    precision_recall(yTest, yScoreDT, yScoreRF, yScoreGB, yScoreXGB, yScoreCB, yHatDT, yHatRF, yHatGB, yHatXGB, yHatCB)


def dt(xTrain, xTest, yTrain):
    clf = DecisionTreeClassifier(min_samples_leaf=75, max_depth=10, criterion='gini', class_weight={0: 1, 1: 3})
    clf.fit(xTrain, yTrain)
    yHat = clf.predict(xTest)
    yScore = clf.predict_proba(xTest)[:, 1]
    return yHat, yScore


def rf(xTrain, xTest, yTrain):
    clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=50, max_depth=10, criterion='entropy',
                                 max_features=0.5, class_weight={0: 1, 1: 3})
    clf.fit(xTrain, yTrain)
    yHat = clf.predict(xTest)
    yScore = clf.predict_proba(xTest)[:, 1]
    return yHat, yScore


def gb(xTrain, xTest, yTrain):
    clf = GradientBoostingClassifier(n_estimators=5, max_features=0.5, min_samples_leaf=1, max_depth=261,
                                     criterion='mse', loss='deviance', learning_rate=1.0)
    clf.fit(xTrain, yTrain)
    yHat = clf.predict(xTest)
    yScore = clf.predict_proba(xTest)[:, 1]
    return yHat, yScore


def xgb(xTrain, xTest, yTrain):
    clf = XGBClassifier(scale_pos_weight=3, n_estimators=50, max_depth=4, reg_lambda=1, gamma=0.1, eta=0.25, alpha=0)
    clf.fit(xTrain, yTrain)
    yHat = clf.predict(xTest)
    yScore = clf.predict_proba(xTest)[:, 1]
    return yHat, yScore


def cb(xTrain, xTest, yTrain):
    clf = CatBoostClassifier(scale_pos_weight=3, max_depth=4, learning_rate=0.25, iterations=100)
    clf.fit(xTrain, yTrain)
    yHat = clf.predict(xTest)
    yScore = clf.predict_proba(xTest)[:, 1]
    return yHat, yScore


if __name__ == "__main__":
    main()
