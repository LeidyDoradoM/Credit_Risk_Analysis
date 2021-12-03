# Credit Risk Analysis

In this week's project we will apply machine learning algorithms to solve a credit card risk problem.  This kind of problem is called imbalanced problem because there are much more credit cards with a low risk of credit compared with the ones with high risk. Therefore, we are going to use `imbalanced-learn` and `scikit-learn` libraries to build and evaluate classification models and sampling algorithms.

## Resampling Models to Predict Credit Risk

An imbalanced dataset is a dataset where each output class is represented by different number of input samples. In these cases is necessary to make a process for balancing the number of samples before classifying the dataset. Here, we are going to test four sampling models: Two oversampling methods: `RandomOverSampler` and `SMOTE`, one undersampling method: `ClusterCentroids`, and `SMOTEENN` that is a mixture between overversampling and undersampling.

The sampling and classification code in *Python* are shown [here](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/credit_risk_resampling.ipynb) and the comparison between the sampling models is performed based on three classification metrics:  **Accuracy**, **Confusion Matrix** and **Precision/Recall**. For all four cases, the same classifier: **Logistic Regression** is used.

### Oversampling Algorithm: Random Oversampler:



![random](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/RandomSampling.png) 

## Oversampling Algorithm: SMOTE Algorithm:

![smote](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/SMOTESampling.png)

### Undersampling Algorithm: Cluster Centroids:

![cluster](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/ClusterCentroid.png)

### Over and Under sampling Algorithm: SMOTEENN:
![smoteen](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/SMOTEENNSampling.png)

## Ensemble Classifiers to Predict Credit Risk

In addition to the sampling and classification approach, we test two ensemble classifiers that resample the dataset and calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report. The code can be found in [here](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/credit_risk_ensambling.ipynb).

### Balance Random Forest Classifier:
![forest](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/RandomForest.png)
### Easy Ensemble Classifier:
![adaboost](https://raw.githubusercontent.com/LeidyDoradoM/Credit_Risk_Analysis/main/Images/AdaBoost.png)

## Sumary

