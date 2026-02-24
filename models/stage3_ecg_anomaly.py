import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import entropy, skew, kurtosis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ===============================
# 0. Ensure Output Folder Exists
# ===============================
os.makedirs("outputs", exist_ok=True)

# ===============================
# 1. Load ECG Data Safely
# ===============================
print("Loading ECG dataset...")

df = pd.read_csv(
    "data/ecg_timeseries.csv",
    header=None,
    low_memory=False
)

print("Original Shape:", df.shape)

# ===============================
# 2. Clean Corrupted Values (CRITICAL FIX)
# ===============================
print("Cleaning corrupted numeric values...")

# Convert everything safely to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Replace invalid values (NaN) with 0
df = df.fillna(0)

# Remove rows fully zero
df = df.loc[~(df == 0).all(axis=1)]

print("Shape after cleaning:", df.shape)

# Convert to float32 to reduce memory
df = df.astype(np.float32)

# ===============================
# 3. Feature Extraction (Robust)
# ===============================

def extract_features(row):
    row = row.values

    # If row is empty or constant, handle safely
    if np.all(row == 0):
        return [0]*11

    # Time domain features
    mean_val = np.mean(row)
    std_val = np.std(row)
    max_val = np.max(row)
    min_val = np.min(row)
    rms = np.sqrt(np.mean(row**2))
    energy = np.sum(row**2)
    skewness = skew(row)
    kurt_val = kurtosis(row)
    peak_to_peak = max_val - min_val

    # Frequency domain
    fft_vals = np.fft.fft(row)
    fft_power = np.abs(fft_vals)

    dominant_freq = np.argmax(fft_power)
    spectral_entropy = entropy(fft_power + 1e-8)

    return [
        mean_val,
        std_val,
        max_val,
        min_val,
        rms,
        energy,
        skewness,
        kurt_val,
        peak_to_peak,
        dominant_freq,
        spectral_entropy
    ]

print("Extracting features...")

features = df.apply(extract_features, axis=1)

feature_df = pd.DataFrame(features.tolist(), columns=[
    "mean", "std", "max", "min",
    "rms", "energy",
    "skewness", "kurtosis",
    "peak_to_peak",
    "dominant_freq",
    "spectral_entropy"
])

print("Feature shape:", feature_df.shape)

# ===============================
# 4. Scale Features
# ===============================
print("Scaling features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

# ===============================
# 5. Train Isolation Forest
# ===============================
print("Training Isolation Forest...")

model = IsolationForest(
    n_estimators=300,
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

# ===============================
# 6. Anomaly Scores
# ===============================
scores = model.decision_function(X_scaled)
anomaly_labels = model.predict(X_scaled)

feature_df["anomaly_score"] = scores
feature_df["anomaly_label"] = anomaly_labels

# ===============================
# 7. Save Model & Outputs
# ===============================
print("Saving outputs...")

joblib.dump(model, "outputs/stage3_ecg_model.pkl")
joblib.dump(scaler, "outputs/stage3_ecg_scaler.pkl")

feature_df.to_csv("outputs/stage3_ecg_features.csv", index=False)

# ===============================
# 8. Visualization
# ===============================
plt.figure()
plt.hist(scores, bins=25)
plt.title("ECG Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.savefig("outputs/stage3_anomaly_distribution.png")
plt.close()

print("Stage 3 ECG Anomaly Model Completed Successfully.")
print("All outputs saved inside /outputs folder.")
