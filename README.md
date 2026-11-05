# forest-cover-type-classification
Forest cover type classification on the Covertype dataset (581,012 samples, 54 features, 7 classes) using Random Forest, Linear SVM, KNN and XGBoost, with a hybrid feature selection method combining Random Forest importance and Mutual Information.
# Forest Cover Type Classification

A comparative study of machine learning algorithms using hybrid feature selection,
developed as a Pattern Recognition project at IIIT Sri City (Department of
Electronics and Communication Engineering, 2026).

## Overview
This project predicts one of seven forest cover types from 54 cartographic
attributes using the Covertype dataset (OpenML, version 3; 581,012 observations).
Four supervised classifiers are compared: Random Forest, Linear SVM, KNN and
XGBoost, each trained on all 54 features and on a reduced set of 10 features.

## Hybrid Feature Selection
Features are ranked using a weighted combination of normalized Random Forest
importance and Mutual Information scores:

    S_hybrid = α · S_RF + (1 − α) · S_MI,   with α = 0.5

The ten highest-ranked features are selected: Elevation,
Horizontal_Distance_To_Roadways, Horizontal_Distance_To_Fire_Points,
Wilderness_Area4, Horizontal_Distance_To_Hydrology, Wilderness_Area1,
Hillshade_9am, Vertical_Distance_To_Hydrology, Soil_Type10 and Aspect.

## Experimental Setup
- Stratified 70:15:15 train / validation / test split (random state 42)
- Standardization of quantitative features for SVM and KNN (scaler fitted on
  training data only)
- Five-fold stratified cross-validation for the selected-feature Random Forest
- Metrics: accuracy, macro precision, macro recall, macro F1, fit time and
  prediction time

## Key Results (Test Set)
| Model | Features | Accuracy (%) | Macro F1 (%) |
|---|---|---|---|
| Random Forest | 10 | 96.43 | 93.31 |
| Random Forest | 54 | 95.69 | 92.53 |
| KNN | 10 | 94.59 | 90.63 |
| KNN | 54 | 93.62 | 89.19 |
| XGBoost | 54 | 80.97 | 74.82 |
| XGBoost | 10 | 79.00 | 71.45 |
| Linear SVM | 54 | 68.23 | 54.39 |
| Linear SVM | 10 | 63.46 | 47.27 |

Random Forest with the 10 selected features performed best while using 81.48%
fewer features. Feature selection helped Random Forest and KNN, but Linear SVM
and XGBoost performed better with all 54 features.

## Tech Stack
Python, scikit-learn, XGBoost, pandas, NumPy, Matplotlib

## Repository Contents
- `main.py` – complete experiment pipeline
- `results/` – saved result CSV files
- `figures/` – generated plots (accuracy, F1, confusion matrix, feature ranking, etc.)
- `Forest_Cover_Type_Report.docx` – full project report

## Usage
    pip install scikit-learn xgboost pandas numpy matplotlib
    python main.py

## Authors
- Suru SaiSreekar
- P. Rahul

Department of Electronics and Communication Engineering, IIIT Sri City
