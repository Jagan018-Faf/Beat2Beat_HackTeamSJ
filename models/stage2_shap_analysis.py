import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("outputs/stage2_model.pkl")
scaler = joblib.load("outputs/stage2_scaler.pkl")

# Load dataset
df = pd.read_csv("data/heart_processed.csv")

X = df.drop("HeartDisease", axis=1)
X_scaled = scaler.transform(X)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# Summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("outputs/stage2_shap_summary.png")
plt.close()

print("SHAP Summary Plot Saved.")
