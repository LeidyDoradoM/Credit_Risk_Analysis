# Credit Risk Analysis

In this week's project we will apply machine learning algorithms to solve a credit card risk problem.  This kind of problem is called imbalanced problem because there are much more credit cards with a low risk of credit compared with the ones with high risk. Therefore, we are going to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate classification models and sampling algorithms.

## Resampling Models to Predict Credit Risk

An imbalanced dataset is a dataset where each output class is represented by different number of input samples. In these cases is necessary to make a process for balancing the number of samples before classifying the dataset. Here, we are going to test four sampling models: Two oversampling methods: `RandomOverSampler` and `SMOTE`, one undersampling method: `ClusterCentroids`, and `SMOTEENN` that is a mixture between overversampling and undersampling.

The sampling and classification code in *Python* are shown [here](https://github.com/LeidyDoradoM/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb) and the comparison between the sampling models is performed based on three classification metrics:  **Accuracy**, **Confusion Matrix** and **Precision/Recall** rates. For all four cases, the same classifier: **Logistic Regression** is used.

### Oversampling Algorithm: Random Oversampler:

Because our interest is to compare how different classification methods perform at identifying high risk credit cards, we are going to focuse on Precision  and Recall rates for the **high risk** class.

![random](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/RandomSampling.png) 

1. **Balance Accuracy:** 67.21%
2. **Precision:** 0.01
3. **Recall:** 0.72

## Oversampling Algorithm: SMOTE Algorithm:

![smote](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/SMOTESampling.png)

1. **Balance Accuracy:** 64.17%
2. **Precision:** 0.01
3. **Recall:** 0.60

### Undersampling Algorithm: Cluster Centroids:

![cluster](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/ClusterCentroid.png)

1. **Balance Accuracy:** 57.59%
2. **Precision:** 0.01
3. **Recall:** 0.59

### Over and Under sampling Algorithm: SMOTEENN:
![smoteen](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/SMOTEENNSampling.png)

1. **Balance Accuracy:** 65.44%
2. **Precision:** 0.01
3. **Recall:** 0.74

## Ensemble Classifiers to Predict Credit Risk

In addition to the sampling and classification approach, we test two ensemble classifiers that resample the dataset and classify the dataset.  As the previous four cases, we calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report. The code can be found in [here](https://github.com/LeidyDoradoM/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb).

### Balance Random Forest Classifier:
![forest](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/RandomForest.png)

1. **Balance Accuracy:** 78.85%
2. **Precision:** 0.03
3. **Recall:** 0.70

### Easy Ensemble Classifier:
![adaboost](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/AdaBoost.png)

1. **Balance Accuracy:** 93.17%
2. **Precision:** 0.09
3. **Recall:** 0.92

## Sumary

As it has been shown in the previous results, the four sampling models and the logistic regression classifier underperformed respect to the ensemble classifiers: *Random Forest* and *Easy Ensemble, Adaboost*.  Especially, **Adaboost** is the best classifier with an overall accuracy of: 93.17%, and with the best recall and precision rates for the high risk class. 

The precision-recall tradeoff in an imbalanced classification problem is very important and challenging. Precision or recall should be prioritized depending on the kind of problem.  In our case, precision should be prioritized since we need a high measure of identification of the high_risk class. And although, Adaboost has the highest precision rate, this value is very small (rate range is 0-1). The model definitely must be improved before to be recomended.
