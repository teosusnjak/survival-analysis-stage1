# Survival Modelling in Residential Aged Care using Explainable AI

Machine learning models developed in our research study ([Link to the document](https://arxiv.org/abs/2312.00271)
) on survival modelling in residential aged care facilities.

## Overview

This repository presents the machine learning models developed in our research study and paper titled "Towards Clinical Prediction with Transparency: An Explainable AI Approach to Survival Modelling in Residential Aged Care". We have applied advanced machine learning techniques to create an interpretable survival model for older people admitted to residential aged care. Our study uses a dataset of 11,944 residents from 40 individual care facilities, spanning data from July 2017 to August 2023.

## Objective

Our goal is to provide a transparent, accurate estimate of survival probabilities for individuals in residential aged care facilities. This can assist in making informed decisions about medical care, particularly towards the end of life.

## TRIPOD Reporting Compliance

The study adheres to the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guideline, ensuring methodological robustness and transparency in reporting. As such, the full model and its example usage is provided here and can be accessed through the Jupyter Notebook.

## Overview of Machine Learning Models Employed in the Study

| Model                         | Characteristics                                                                                                                                                                                                                                       |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cox Proportional Hazards | The CoxPH is an extension of the classical Cox Proportional Hazards model, containing a penalty term to better manage high-dimensional datasets. Its core advantage is its capacity to manage the problems of both high dimensionality and event sparsity. |
| Elastic Net                 | The Elastic Net model amalgamates the L1 and L2 regularization techniques of Lasso and Ridge Regression, respectively. This hybridization allows the model to efficiently navigate the challenges of multicollinearity and variable selection.             |
| Ridge Regression            | Ridge Regression employs L2 regularization to provide an alternative approach to handling multicollinearity. It is adept at shrinking coefficients, stabilizing them in the presence of highly correlated variables.                                  |
| Lasso                 | Lasso utilizes L1 regularization to achieve both regularization and variable selection. It is especially useful for high-dimensional datasets where feature selection is vital, as it drives some coefficients to zero.                                |
| Gradient Boosting      | Gradient Boosting is a state-of-the-art ensemble learning technique that builds strong predictive models by aggregating weak learners. Its adaptability and effectiveness have been empirically validated in many settings, including healthcare.          |
| XGBoost                      | XGBoost is as an optimized variant of the Gradient Boosting algorithm, notable for computational efficiency and scalability. Its predictive capability has been demonstrated through its dominance in various ML competitions.                          |
| Random Forest            | Random Forest is an ensemble of decision trees, each constructed with a bootstrapped sample of the data and a subset of variables. Its robustness against outliers and irrelevant features makes it well suited for modelling clinical data.              |


## Overview of Hyperparameter Settings for the Models

| Model             | Hyperparameters                                                                                                                                                                                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CoxPH [tibshirani1997]          | None                                                                                                                                                                                                                                          |
| Elastic Net       | l1_ratio=1.0, n_alphas=1, alphas=[0.00034], normalize=True, fit_baseline_model=True                                                                                                                                                          |
| Ridge Regression  | l1_ratio=$10^{-100}$, n_alphas=1, alphas=[2.24e-06], normalize=True, fit_baseline_model=True                                                                                                                                                 |
| Lasso             | l1_ratio=0.9, alpha_min_ratio=0.01, fit_baseline_model=True                                                                                                                                                                                   |
| Gradient Boosting | n_estimators=771, min_samples_split=20.04, max_depth=7, min_samples_leaf=1.85, learning_rate=0.28, dropout_rate=0.05, objective='survival:cox', max_features=4, subsample=0.83                                                               |
| XGBoost           | num_boost_round=1107, learning_rate=0.018, max_depth=3, colsample_bytree=0.83, gamma=0.49, objective='survival:cox', subsample=0.58                                                                                                           |
| Random Forest     | n_estimators=592, min_samples_split=2.54, max_depth=7, min_samples_leaf=20.89                                                                                                                                                                 |

## Repository Structure

- `models/`: Contains the trained machine learning models (XGBoost and Gradient Boosting), including the calibration  model (Logistic Regression).
- `datasets/`: Contains example data points representing hypothetical patients used in the demo notebook.
- `notebooks/`: Jupyter notebook demonstrating the model usage and SHAP analysis.

## Python Libraries

The following libraries are needed:

- scikit-survival (version 0.21.0)
- XGBoost (version 1.7.6)

## Contact

For queries regarding this project, please refer to the contact details in the study paper ([Link to the document](https://arxiv.org/abs/2312.00271)).
