import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ===============================
# 1. Load ECG Data
# ===============================
print("Loading ECG dataset...")

df = pd.read_excel("data/ecg_timeseries.xlsx", header=None)

# Remove rows that are fully zero
df = df.loc[~(df == 0).all(axis=1)]

print("Shape after removing zero rows:", df.shape)

# ===============================
# 2. Feature Extraction
# ===============================

def extract_features(row):
    row = np.array(row)

    mean_val = np.mean(row)
    std_val = np.std(row)
    max_val = np.max(row)
    min_val = np.min(row)
    rms = np.sqrt(np.mean(row**2))
    energy = np.sum(row**2)

    # Frequency domain
    fft_vals = np.fft.fft(row)
    fft_power = np.abs(fft_vals)

    dominant_freq = np.argmax(fft_power)
    spectral_entropy = entropy(fft_power)

    return [
        mean_val,
        std_val,
        max_val,
        min_val,
        rms,
        energy,
        dominant_freq,
        spectral_entropy
    ]

print("Extracting features...")

features = df.apply(extract_features, axis=1)
feature_df = pd.DataFrame(features.tolist(), columns=[
    "mean", "std", "max", "min",
    "rms", "energy",
    "dominant_freq", "spectral_entropy"
])

print("Feature shape:", feature_df.shape)

# ===============================
# 3. Scale Features
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

# ===============================
# 4. Train Isolation Forest
# ===============================
print("Training Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

model.fit(X_scaled)

# Anomaly scores
scores = model.decision_function(X_scaled)
anomaly_labels = model.predict(X_scaled)

feature_df["anomaly_score"] = scores
feature_df["anomaly_label"] = anomaly_labels

# ===============================
# 5. Save Model & Scaler
# ===============================
joblib.dump(model, "outputs/stage3_ecg_model.pkl")
joblib.dump(scaler, "outputs/stage3_ecg_scaler.pkl")

feature_df.to_csv("outputs/stage3_ecg_features.csv", index=False)

print("Stage 3 ECG Anomaly Model Completed.")

# ===============================
# 6. Visualization
# ===============================
plt.figure()
plt.hist(scores, bins=20)
plt.title("ECG Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.savefig("outputs/stage3_anomaly_distribution.png")
