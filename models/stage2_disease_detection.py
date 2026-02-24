import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ===============================
# 0. Ensure Output Folder Exists
# ===============================
os.makedirs("outputs", exist_ok=True)

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/heart_processed.csv")

print("Stage 2 Dataset Loaded Successfully")
print("Shape:", df.shape)

# ===============================
# 2. Features & Target
# ===============================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Scaling (ADDED FIX)
# ===============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Train Model
# ===============================
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ===============================
# 6. Evaluation
# ===============================
y_preds = model.predict(X_test_scaled)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)

print("\nStage 2 ROC-AUC:", roc_auc)
print("\nClassification Report:")
print(classification_report(y_test, y_preds))

# ===============================
# 7. ROC Curve
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stage 2 ROC Curve")
plt.savefig("outputs/stage2_roc_curve.png")
plt.close()

# ===============================
# 8. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stage 2 Confusion Matrix")
plt.savefig("outputs/stage2_confusion_matrix.png")
plt.close()

# ===============================
# 9. Feature Importance
# ===============================
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
plt.barh(feature_names, importances)
plt.title("Stage 2 Feature Importance")
plt.tight_layout()
plt.savefig("outputs/stage2_feature_importance.png")
plt.close()

# ===============================
# 10. Save Model & Scaler (CRITICAL FIX)
# ===============================
joblib.dump(model, "outputs/stage2_model.pkl")
joblib.dump(scaler, "outputs/stage2_scaler.pkl")

print("\nStage 2 Model & Scaler Saved Successfully.")
print("Stage 2 Training Completed Successfully.")
