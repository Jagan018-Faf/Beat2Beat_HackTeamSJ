import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/cardio_base.csv", sep=";")
print("Columns in dataset:", df.columns)

# ===============================
# 2. Basic Preprocessing
# ===============================

# Convert age from days to years
df["age"] = df["age"] / 365

# Remove unrealistic blood pressure values
df = df[(df["ap_hi"] > 50) & (df["ap_hi"] < 250)]
df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 150)]

# Define features and target
X = df.drop("cardio", axis=1)
y = df["cardio"]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Train Models
# ===============================

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ===============================
# 6. Evaluation
# ===============================

log_preds = log_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)

print("Logistic Regression ROC-AUC:",
      roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:,1]))

print("Random Forest ROC-AUC:",
      roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

print("\nClassification Report (Logistic Regression)")
print(classification_report(y_test, log_preds))

from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

# ===============================
# ROC Curve
# ===============================
y_probs = log_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stage 1 ROC Curve")
plt.savefig("outputs/stage1_roc_curve.png")

# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, log_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stage 1 Confusion Matrix")
plt.savefig("outputs/stage1_confusion_matrix.png")

# ===============================
# 7. Feature Importance (RF)
# ===============================
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure()
plt.barh(feature_names, importances)
plt.title("Stage 1 Feature Importance")
plt.tight_layout()
plt.savefig("outputs/stage1_feature_importance.png")

# ===============================
# 8. Save Model
# ===============================
joblib.dump(rf_model, "outputs/stage1_model.pkl")
joblib.dump(scaler, "outputs/stage1_scaler.pkl")

print("Stage 1 Model Training Completed.")
