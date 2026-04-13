# Disease-Predicting-Model
Machine Learning model for liver disease prediction
## Feature Selection

Feature selection was performed based on domain knowledge and correlation analysis.

Initially, multiple biochemical parameters were available in the dataset. A correlation heatmap was used to understand relationships between features and identify multicollinearity.

### Key Observations:

* High correlation between:

  * Total Bilirubin and Direct Bilirubin (0.87)
  * Albumin and A/G Ratio (0.79)
  * SGPT and SGOT (0.78)

### Action Taken:

To reduce redundancy and improve model interpretability:

* One feature from highly correlated pairs was selected
* Less informative or redundant features were removed

### Final Selected Features:

* Age
* Gender
* Direct Bilirubin
* Albumin
* SGOT

### Reasoning:

These features were chosen because they:

* Show meaningful correlation with liver disease
* Reduce multicollinearity
* Improve model stability and generalization

This approach ensures a balance between model performance and interpretability.

