import joblib
import numpy as np
import pandas as pd

# ===============================
# Load Models & Scalers
# ===============================
stage1_model = joblib.load("outputs/stage1_model.pkl")
stage1_scaler = joblib.load("outputs/stage1_scaler.pkl")

stage2_model = joblib.load("outputs/stage2_model.pkl")
stage2_scaler = joblib.load("outputs/stage2_scaler.pkl")

stage3_model = joblib.load("outputs/stage3_ecg_model.pkl")
stage3_scaler = joblib.load("outputs/stage3_ecg_scaler.pkl")

print("All models loaded successfully.")

# ===============================
# Stage 1 SAFE INPUT
# ===============================
stage1_columns = stage1_scaler.feature_names_in_

stage1_data = pd.DataFrame(
    np.zeros((1, len(stage1_columns))),
    columns=stage1_columns
)

stage1_scaled = stage1_scaler.transform(stage1_data)
stage1_risk = stage1_model.predict_proba(stage1_scaled)[0][1]

# ===============================
# Stage 2 SAFE INPUT
# ===============================
stage2_columns = stage2_scaler.feature_names_in_

stage2_data = pd.DataFrame(
    np.zeros((1, len(stage2_columns))),
    columns=stage2_columns
)

stage2_scaled = stage2_scaler.transform(stage2_data)
stage2_risk = stage2_model.predict_proba(stage2_scaled)[0][1]

# ===============================
# Stage 3 SAFE INPUT
# ===============================
ecg_data = np.zeros((1, stage3_scaler.n_features_in_))

ecg_scaled = stage3_scaler.transform(ecg_data)
ecg_score = stage3_model.decision_function(ecg_scaled)[0]

# Convert anomaly score to positive risk-like scale
ecg_risk = abs(ecg_score)

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
