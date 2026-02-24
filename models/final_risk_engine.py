import joblib
import numpy as np
import pandas as pd

# ===============================
# Load Models
# ===============================

stage1_model = joblib.load("outputs/stage1_model.pkl")
stage1_scaler = joblib.load("outputs/stage1_scaler.pkl")

stage2_model = joblib.load("outputs/stage2_model.pkl")
stage2_scaler = joblib.load("outputs/stage2_scaler.pkl")

stage3_model = joblib.load("outputs/stage3_ecg_model.pkl")
stage3_scaler = joblib.load("outputs/stage3_ecg_scaler.pkl")

print("All models loaded successfully.")

# ===============================
# Example Patient Input
# ===============================

# ---- Stage 1 input ----
stage1_input = np.array([[45, 170, 70, 120, 80, 1, 0, 0, 1]])

stage1_scaled = stage1_scaler.transform(stage1_input)
stage1_risk = stage1_model.predict_proba(stage1_scaled)[0][1]

# ---- Stage 2 input ----
stage2_input = np.array([[50, 130, 250, 0, 150, 1, 0, 0, 0, 0, 1, 0]])

stage2_scaled = stage2_scaler.transform(stage2_input)
stage2_risk = stage2_model.predict_proba(stage2_scaled)[0][1]

# ---- Stage 3 input ----
# Simulated ECG feature vector (11 features)
ecg_features = np.random.rand(1, 11)

ecg_scaled = stage3_scaler.transform(ecg_features)
ecg_score = stage3_model.decision_function(ecg_scaled)[0]

# Convert anomaly score to 0–1 scale
ecg_risk = 1 - (ecg_score - ecg_score.min()) if isinstance(ecg_score, np.ndarray) else 1 - ecg_score

# ===============================
# Final Risk Fusion
# ===============================

final_risk_score = (
    0.3 * stage1_risk +
    0.4 * stage2_risk +
    0.3 * ecg_risk
)

print("\n===== FINAL CARDIAC RISK REPORT =====")
print("Stage 1 Lifestyle Risk:", round(stage1_risk, 3))
print("Stage 2 Clinical Risk:", round(stage2_risk, 3))
print("Stage 3 ECG Risk:", round(float(ecg_risk), 3))
print("Final Integrated Risk Score:", round(float(final_risk_score), 3))
