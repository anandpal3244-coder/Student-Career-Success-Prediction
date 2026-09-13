# ================================================================
# TRAIN_MODEL.PY
# Student Career Success Prediction - FINAL TRAINING PIPELINE
# ================================================================
# Run this file ONCE in your PyCharm project.
# It creates all .pkl files required by app.py.
# ================================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(r"C:\Data_Set\student_career_success_dataset.csv")
LOCAL_DATA_PATH = BASE_DIR / "student_career_success_dataset.csv"

if DATA_PATH.exists():
    data_path = DATA_PATH
elif LOCAL_DATA_PATH.exists():
    data_path = LOCAL_DATA_PATH
else:
    raise FileNotFoundError(
        "Dataset not found. Put student_career_success_dataset.csv "
        "in the project folder or update DATA_PATH in this file."
    )

print(f"Using dataset: {data_path}")
print(f"scikit-learn version: {sklearn.__version__}")

df = pd.read_csv(data_path)
df = df.drop_duplicates().copy()

# Target encoding
df["Placement_Status"] = df["Placement_Status"].map({
    "Placed": 1,
    "Not Placed": 0
})

if df["Placement_Status"].isna().any():
    raise ValueError("Placement_Status contains unexpected values.")

# Keep the same prediction columns as the project notebook.
drop_cols = [
    "Student_ID",
    "Company_Tier",
    "Career_Field",
    "Placement_Mode",
    "Starting_Salary_USD",
    "Placement_Status"
]

X = df.drop(columns=drop_cols)
y = df["Placement_Status"].astype(int)

categorical_cols = [
    "Gender",
    "University_Year",
    "Major",
    "Academic_Performance",
    "GitHub_Profile",
    "Leadership_Experience",
    "LinkedIn_Profile",
    "English_Proficiency"
]

numeric_cols = [c for c in X.columns if c not in categorical_cols]

for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Exact choices are saved so Streamlit never sends category names
# that did not exist during training.
input_options = {
    col: sorted(X[col].dropna().astype(str).unique().tolist())
    for col in categorical_cols
}
joblib.dump(input_options, BASE_DIR / "input_options.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nOriginal target distribution:")
print(y.value_counts(normalize=True).mul(100).round(2))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SkPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_cols
        ),
        (
            "cat",
            SkPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            categorical_cols
        )
    ],
    remainder="drop"
)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.5, solver="liblinear", max_iter=2000
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=5, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
    )
}

results = []

print("\nTraining and evaluating models...")
for name, estimator in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", estimator)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }
    results.append(row)
    print(
        f"{name}: Accuracy={row['Accuracy']:.4f}, "
        f"F1={row['F1 Score']:.4f}, AUC={row['ROC-AUC']:.4f}"
    )

comparison_df = pd.DataFrame(results).sort_values(
    "F1 Score", ascending=False
).reset_index(drop=True)
comparison_df.to_pickle(BASE_DIR / "model_comparison.pkl")

# Select the best model by F1.
best_name = comparison_df.loc[0, "Model"]
print(f"\nSelected base model: {best_name}")

best_estimator = models[best_name]
base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", best_estimator)
])

# Calibration is deliberate: the app displays a probability, not only a 0/1 class.
# It prevents a model with overconfident raw probabilities from making almost every
# student look like a 99-100% placement case.
final_model = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=3,
    n_jobs=-1
)

print("Training final calibrated model...")
final_model.fit(X_train, y_train)

final_pred = final_model.predict(X_test)
final_prob = final_model.predict_proba(X_test)[:, 1]

final_accuracy = accuracy_score(y_test, final_pred)
final_precision = precision_score(y_test, final_pred, zero_division=0)
final_recall = recall_score(y_test, final_pred, zero_division=0)
final_f1 = f1_score(y_test, final_pred, zero_division=0)
final_auc = roc_auc_score(y_test, final_prob)

model_info = {
    "model_name": f"Calibrated {best_name}",
    "accuracy": final_accuracy,
    "precision": final_precision,
    "recall": final_recall,
    "f1_score": final_f1,
    "roc_auc": final_auc
}

print("\nFINAL MODEL RESULTS")
for k, v in model_info.items():
    print(f"{k}: {v if k == 'model_name' else f'{v:.4f}'}")

# Save the final model as ONE complete object.
# This contains preprocessing + SMOTE + classifier + calibration.
joblib.dump(final_model, BASE_DIR / "student_placement_model.pkl")
joblib.dump(model_info, BASE_DIR / "model_info.pkl")
joblib.dump(X_train.columns.tolist(), BASE_DIR / "feature_names.pkl")

# Confusion matrix and classification report
joblib.dump(
    confusion_matrix(y_test, final_pred),
    BASE_DIR / "confusion_matrix.pkl"
)

report_df = pd.DataFrame(
    classification_report(y_test, final_pred, output_dict=True, zero_division=0)
).transpose()
report_df.to_pickle(BASE_DIR / "classification_report.pkl")

# Feature importance for the selected base model.
# Fit a separate copy on the full training data for transparent feature reporting.
importance_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", best_estimator)
])
importance_pipeline.fit(X_train, y_train)

pre = importance_pipeline.named_steps["preprocessor"]
model_obj = importance_pipeline.named_steps["model"]
feature_names_out = pre.get_feature_names_out()

if hasattr(model_obj, "coef_"):
    importance_df = pd.DataFrame({
        "Feature": feature_names_out,
        "Coefficient": model_obj.coef_[0]
    })
    importance_df["Absolute_Importance"] = importance_df["Coefficient"].abs()
elif hasattr(model_obj, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": feature_names_out,
        "Importance": model_obj.feature_importances_
    })
    importance_df["Absolute_Importance"] = importance_df["Importance"]
else:
    importance_df = pd.DataFrame({
        "Feature": feature_names_out,
        "Importance": 0.0,
        "Absolute_Importance": 0.0
    })

importance_df = importance_df.sort_values(
    "Absolute_Importance", ascending=False
).reset_index(drop=True)
importance_df.to_pickle(BASE_DIR / "feature_importance.pkl")

# Save extra reproducibility information
metadata = {
    "sklearn_version": sklearn.__version__,
    "selected_base_model": best_name,
    "prediction_columns": X.columns.tolist(),
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols
}
joblib.dump(metadata, BASE_DIR / "training_metadata.pkl")

print("\n================================================")
print("TRAINING COMPLETE")
print("================================================")
print("Created:")
for filename in [
    "student_placement_model.pkl",
    "model_info.pkl",
    "feature_names.pkl",
    "model_comparison.pkl",
    "feature_importance.pkl",
    "confusion_matrix.pkl",
    "classification_report.pkl",
    "input_options.pkl",
    "training_metadata.pkl"
]:
    print(" -", filename)
print("\nTest several very different student profiles in Streamlit.")
