# Cross-Selling Vehicle Insurance to Current Health Insurance Policy Owners

## Contributions
All contents of this project belong to:
* **Isaac Yoo**
* **Tae Yeon Kim**
* **Alexandru Rudi**

## Abstract

This paper aims to understand consumer behavior in the cross-selling of vehicle insurance to current health insurance policy owners. Our dataset consists of features that detail the vehicular history of numerous individuals, with the label being their interest in vehicle insurance from their health insurance provider. The dataset is preprocessed to improve interpretability and account for random noise. Our models consist of the Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, CatBoost, and XGBoost classifiers. Hyperparameter tuning is done through the K-Fold Cross-Validation technique. Due to a large imbalance within the labels, optimizations are made for the F1 score rather than accuracy/area under the curve. Our best performing model was CatBoost, obtaining an F1 score of 0.455 while maintaining a solid AUC of 0.856. We have obtained very similar performance from XGBoost and Random Forest, and even the simple Decision Tree model ended up being very competitive, obtaining an F1 score of 0.444. These results are on par with the other results on Kaggle, where the largest F1 value obtained is 0.459, using XGBoost.

## Built With

* Python
* Scikit-Learn

## Models
* Decision Tree
* Random Forest
* Gradient Boosting
* Logistic Regression
* CatBoost
* XGBoost Classifiers

## Contents
* Data Cleanup / Preprocessing
* Models, as per list above
* Evaluation

## Date

December 2020
