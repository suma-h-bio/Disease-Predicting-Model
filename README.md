# Disease-Predicting-Model

Machine Learning model for liver disease prediction using multiple algorithms and optimization techniques.

---

## Dataset Overview

The dataset contains clinical and biochemical parameters of liver patients.

### Summary Statistics:

* Total Samples: 583
* Features: 10
* Target: Disease (1 = Disease, 0 = Healthy)

### Feature Statistics (Mean Values):

**Feature                     Mean **  
Age                   -     44.74  
Gender                -     0.75   
Total Bilirubin       -     3.29   
Direct Bilirubin      -     1.48   
Total Protein         -     290.57 
Albumin               -     80.71  
A/G Ratio             -     109.91 
SGPT                  -     6.48   
SGOT                  -     3.14   
Alkaline Phosphatase  -     0.94   


## Feature Analysis

A correlation heatmap was used to understand relationships between features.

### Updated Strategy:

Instead of removing correlated features, **all features were retained** because:

* Tree-based models (Random Forest, XGBoost) handle multicollinearity well
* More features improved predictive performance
* Helps preserve biological/clinical relevance

## Machine Learning Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost
* Naive Bayes
* XGBoost

## Model Optimization Techniques

### 1. Data Preprocessing

* Gender encoding (Male = 1, Female = 0)
* Missing value imputation using median


### 2. Feature Scaling

* StandardScaler applied where required

### 3. Handling Imbalanced Data

* SMOTE (Synthetic Minority Oversampling Technique) used

### 4. Pipeline Implementation (Important)

* Combined SMOTE + Scaling + Model in pipeline
* Prevents data leakage


### 5. Regularization Techniques

* max_depth → controls tree complexity
* min_samples_split → prevents overfitting

### 6. Cross-Validation

* Stratified K-Fold (k=5)
* Ensures model robustness

### 7. Threshold Tuning (Key Improvement )

Instead of using default threshold (0.5):

* Optimized threshold for each model
* Reduced false negatives significantly
* Improved recall (important in medical diagnosis)

## Key Insight

> Threshold tuning had the biggest impact on performance compared to model selection.

## Conclusion

* Ensemble models performed better than simple models
* Threshold tuning is critical in medical ML problems
* Model prioritizes **minimizing missed disease cases (FN)**

## Project Structure

* `data/` → dataset files
* `notebooks/` → experiments
* `src/` → model scripts
* `liver_disease.py` → main pipeline
* `requirements.txt` → dependencies

## Code Reference

Main implementation: 

## Author

Suma
(Bioinformatics & Machine Learning Enthusiast)
