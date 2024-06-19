# Prudential Life Insurance Risk Assessment

This project aims to analyze and develop a predictive model for Prudential Life Insurance to improve the accuracy of risk assessment and reduce the time required to get an insurance quote.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing Steps](#preprocessing-steps)
4. [Data Analysis Procedure](#data-analysis-procedure)
5. [Base Models](#base-models)
6. [Feature Engineering](#feature-engineering)
7. [Feature Selection](#feature-selection)
8. [Hyperparameter Tuning and Validation](#hyperparameter-tuning-and-validation)
9. [Train and Predict](#train-and-predict)
10. [Model Interpretation](#model-interpretation)
11. [Suggested Improvements](#suggested-improvements)
12. [Conclusion](#conclusion)

## Overview

The objective of this project is to build a predictive model to enhance risk assessment for Prudential Life Insurance. The model focuses on improving accuracy and efficiency in providing insurance quotes.

## Dataset

The dataset for this project is sourced from Kaggle's Prudential Life Insurance Assessment competition. The dataset can be found [here](https://www.kaggle.com/c/prudential-life-insurance-assessment/overview).

## Preprocessing Steps

1. Convert text data to lowercase and strip extra spaces.
2. Remove duplicated rows and empty rows/columns.
3. Split the data to avoid leakage.
4. Handle missing values considering data distribution (skewness, multimodal, normal, categorical).
5. Remove irrelevant features using domain knowledge.
6. Reduce memory usage by reassigning data types based on min-max values.

## Data Analysis Procedure

1. Convert categorical dtype (Product_Info_2) to numerical.
2. Select the top 10 features that highly influence the target variable using mutual information.
3. Analyze feature correlation using a correlation matrix and heatmap.
4. Analyze feature interactions with the target response.
5. Analyze the imbalance in the target variable.
6. Perform cluster analysis using Principal Component Analysis (PCA) and KMeans.

## Base Models

1. **MLPClassifier**: 
    - Evaluate performance based on accuracy and F1 score.
2. **RandomForestClassifier**:
    - Evaluate performance based on accuracy and F1 score.

## Feature Engineering

1. Apply Target Encoding and evaluate.
2. Apply Frequency Encoding and evaluate.
3. Bin continuous variables and evaluate each.
4. Generate new features from feature interaction and evaluate.
5. Encode categorical data and evaluate.

## Feature Selection

Top features are selected after feature engineering using both variance and mutual information.

## Hyperparameter Tuning and Validation

1. Hyperparameter tuning of tree-based algorithms (RF, XGB, CatBoost) using F1 score for evaluation.
2. Apply SMOTEENN to address class imbalance.
3. Use StratifiedKFold to distribute the target variable evenly.

## Train and Predict

1. Retrain tree-based algorithms with the full training dataset and predict.
2. Train and predict with a stacked model using XGB, RF, and CatBoost as base models and LR as a meta model.
3. Train and predict with a FeedForward Neural Network (FNN) / Multilayer Perceptron (MLP).
4. Compare accuracy and F1 score results.

## Model Interpretation

1. Perform Feature Importance Analysis.
2. Conduct Lift and Gain Analysis.
3. Use LIME Methodology for model interpretation.

## Suggested Improvements

1. Integrate all preprocessing and model selection into a single pipeline.
2. Hypertune the best model as a common practice.
3. Create new features from KMeans clustering results.

## Conclusion

This project demonstrates a comprehensive approach to developing a predictive model for insurance risk assessment, leveraging various machine learning techniques and thorough data analysis.
The project also highlights the practical implications of machine learning in improving business processes, such as risk assessment and quote generation, leading to better decision-making and operational efficiency.
