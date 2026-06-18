# Methodology

## Study Area

Calder Valley, West Yorkshire, United Kingdom.

## Predictor Variables

* DEM
* Slope
* NDVI
* Land Cover
* Rainfall
* Distance to River
* Flow Accumulation

## Machine Learning Model

Random Forest Classifier

Parameters:

* n_estimators = 300
* min_samples_leaf = 2
* class_weight = balanced

## Model Performance

* Accuracy = 84.8%
* ROC-AUC = 0.91
* Flood Recall = 92%

## Outputs

* Flood Susceptibility Map
* ROC Curve
* Confusion Matrix
* Feature Importance Chart