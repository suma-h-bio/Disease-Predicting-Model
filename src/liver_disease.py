# ==============================
# FINAL IMPROVED MODEL (ROBUST + NO OVERFITTING)
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("/content/liver_disease.csv")

# ==============================
# DATA CLEANING
# ==============================

df["gender"] = df["gender"].str.strip().str.lower().map({
    "male": 1,
    "female": 0
})

df["L_disease"] = df["L_disease"].replace({1:1, 2:0})

df = df.fillna(df.median(numeric_only=True))

# ==============================
# FEATURES
# ==============================

selected_features = [
    "age", "gender", "Direct Bilirubin",
    "Albumin", "A/G ratio", "SGPT", "SGOT"
]

X = df[selected_features]
y = df["L_disease"]

# ==============================
# SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# DEFINE MODELS WITH PIPELINE
# ==============================

models = {
    "Logistic Regression": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("model", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),

    "Random Forest": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        ))
    ]),

    "Naive Bayes": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "XGBoost": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("model", XGBClassifier(
            eval_metric='logloss',
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            random_state=42
        ))
    ])
}

# ==============================
# CROSS VALIDATION SETUP
# ==============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================
# TRAIN & EVALUATE
# ==============================

results = []

for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")

    # Cross-validation score (IMPORTANT ADDITION)
    cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()
    print(f"Cross-Validated F1 Score: {cv_score:.3f}")

    # Train on full training set
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = y_pred

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("Confusion Matrix:\n", cm)
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "CV_F1": cv_score,
        "Precision_0": report["0"]["precision"],
        "Recall_0": report["0"]["recall"],
        "Precision_1": report["1"]["precision"],
        "Recall_1": report["1"]["recall"],
        "F1_1": report["1"]["f1-score"],
        "FP": FP,
        "FN": FN
    })

# ==============================
# FINAL RESULTS
# ==============================

print("\nFINAL ROBUST RESULTS")
print(pd.DataFrame(results).sort_values(by="CV_F1", ascending=False))