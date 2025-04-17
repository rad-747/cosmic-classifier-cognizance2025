# Cosmic Classifier 🚀

This repository contains the notebook **Cosmicclassifier.ipynb**, developed as part of **Cognizance 2025**, the Annual Technical Fest of IIT Roorkee.  
The goal of this project was to classify **cosmic entities into 10 categories** using tabular astrophysical data with a CatBoost-powered machine learning pipeline.

##  Project Overview

This notebook builds a robust multi-class classification model using:
- **CatBoostClassifier**
- **7-Fold Cross-Validation**
-  Domain-inspired **feature engineering**
-  Missing value imputation
-  Performance evaluation via precision, recall, and F1-score

##  Feature Engineering

Three custom interaction features were introduced:
- `Gravity_Proximity_Ratio = Gravity / (Proximity to Star + 1)`
- `Water_Density_Product = Water Content * Atmospheric Density`
- `Temp_Gravity_Ratio = Surface Temperature / (Gravity + 1)`

Also included:
- Missing value flags for all numeric columns
- String conversion for categorical variables

## Model Details

- **Model Used:** `CatBoostClassifier`
- **Hyperparameters:**  
  `learning_rate=0.05`, `iterations=1000`, `depth=8`, `l2_leaf_reg=3`
- **Cross-Validation:** `7 folds` using `KFold`

##  Performance Metrics

###  Cross-Validation (Training Set)
- **K-Fold Accuracy:** `0.8939`
- **Macro F1-Score:** ~`0.89`
- **Precision/Recall** across 10 classes: ranges from `0.81` to `0.96`

>  Highlights:  
> - Class 1.0 achieved 96% precision and 98% recall  
> - Most classes consistently above 85% F1-score

###  Test Set Results
- **Test Accuracy:** `0.8891`
- **Macro F1-Score:** `0.89`
- High generalization across all 10 classes (precision and recall mostly above 85%)

##  Installation

```bash
pip install pandas numpy scikit-learn catboost jupyter matplotlib seaborn
