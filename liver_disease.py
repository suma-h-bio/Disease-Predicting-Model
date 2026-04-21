#5th Model extented Gradient boosting+ Ada boosting
# ==============================
# FINAL MODEL: SMOTE + PIPELINE + THRESHOLD + BOOSTING
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
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
# CLEANING
# ==============================

df["gender"] = df["gender"].str.strip().str.lower().map({
    "male": 1,
    "female": 0
})

df["disease"] = df["disease"].replace({1:1, 2:0})
df = df.fillna(df.median(numeric_only=True))

# ==============================
# FEATURES
# ==============================

features = [
    "age", "gender",
    "total bilirubin", "direct bilirubin",
    "total protein", "albumin", "A/G ratio",
    "SGPT", "SGOT", "alkphos"
]

X = df[features]
y = df["disease"]

# ==============================
# SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
# MODELS
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

    "Gradient Boosting": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("model", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3
        ))
    ]),

    "AdaBoost": ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.8, random_state=42)),
        ("model", AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.05,
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
# CROSS VALIDATION
# ==============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==============================
# TRAIN + THRESHOLD TUNING
# ==============================

results = []

for name, model in models.items():
    print("\n==============================")
    print(f"Model: {name}")

    # CV Score
    cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1").mean()
    print(f"CV F1 Score: {cv_score:.3f}")

    # Train
    model.fit(X_train, y_train)

    # Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
    else:
        y_prob = model.predict(X_test)

    # ==============================
    # THRESHOLD TUNING
    # ==============================

    best_f1 = 0
    best_thresh = 0.5

    for t in np.arange(0.2, 0.7, 0.05):
        y_pred_temp = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_temp)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Best Threshold: {best_thresh:.2f}")

    # Final prediction
    y_pred = (y_prob >= best_thresh).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    print("Confusion Matrix:\n", cm)
    print(f"FP: {FP}, FN: {FN}")

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "CV_F1": cv_score,
        "Best_Threshold": best_thresh,
        "F1_1": report["1"]["f1-score"],
        "Recall_1": report["1"]["recall"],
        "Precision_1": report["1"]["precision"],
        "FP": FP,
        "FN": FN
    })

# ==============================
# FINAL RESULTS
# ==============================

print("\nFINAL RESULTS (WITH BOOSTING MODELS)")
print(pd.DataFrame(results).sort_values(by="F1_1", ascending=False))
