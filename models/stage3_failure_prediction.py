import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/cardiac_failure_processed.csv")

print("Stage 3 Dataset Loaded")
print("Shape:", df.shape)
print("Columns:", df.columns)

# ===============================
# 2. Define Features & Target
# ===============================

TARGET_COLUMN = "DEATH_EVENT"   # Change if needed

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Train Model
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5. Evaluation
# ===============================
y_preds = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probs)

print("\nStage 3 ROC-AUC:", roc_auc)
print("\nClassification Report:")
print(classification_report(y_test, y_preds))

# ===============================
# 6. ROC Curve
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stage 3 ROC Curve")
plt.savefig("outputs/stage3_roc_curve.png")

# ===============================
# 7. Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_preds)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stage 3 Confusion Matrix")
plt.savefig("outputs/stage3_confusion_matrix.png")

# ===============================
# 8. SHAP Interpretability
# ===============================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values[1], X_test, show=False)
plt.title("Stage 3 SHAP Summary Plot")
plt.savefig("outputs/stage3_shap_summary.png")

# ===============================
# 9. Save Model
# ===============================
joblib.dump(model, "outputs/stage3_model.pkl")

print("\nStage 3 Model Training Completed.")
