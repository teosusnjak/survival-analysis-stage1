# Survival Modelling in Residential Aged Care using Explainable AI

Machine learning models developed in our research study on survival modelling in residential aged care at Ryman facilities.


## Overview

This repository presents the machine learning models developed in our research study and paper titled "Towards Clinical Prediction with Transparency: An Explainable AI Approach to Survival Modelling in Residential Aged Care". We have applied advanced machine learning techniques to create an interpretable survival model for older people admitted to residential aged care. Our study uses a dataset of 11,944 residents from 40 individual care facilities, spanning data from July 2017 to August 2023.

## Objective

Our goal is to provide a transparent, accurate estimate of survival probabilities for individuals in residential aged care facilities. This can assist in making informed decisions about medical care, particularly towards the end of life.

## TRIPOD Reporting Compliance

The study adheres to the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guideline, ensuring methodological robustness and transparency in reporting. As such, the full model and its example usage is provided here and can be accessed through the Jupyter Notebook.


## Repository Structure

- `models/`: Contains the trained machine learning models (XGBoost and Gradient Boosting), including the calibration  model (Logistic Regression).
- `datasets/`: Contains example data points representing hypothetical patients used in the demo notebook.
- `notebooks/`: Jupyter notebook demonstrating the model usage and SHAP analysis.

## Python Libraries

The following libraries are needed:

- scikit-survival (version 0.21.0)
- XGBoost (version 1.7.6)

## Contact

For queries regarding this project, please refer to the contact details in the study paper.
