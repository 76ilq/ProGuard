# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vBWg7b_CoXqfN4xJ6skfsRa1TVj9Tcze
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------
# 📥 Upload CSV file
# --------------------------------
from google.colab import files
uploaded = files.upload()

# Load dataset
import io
data = pd.read_csv(io.BytesIO(uploaded['objective_injury_data.csv']))

# --------------------------------
# ⚙️ Constants
# --------------------------------
HR_rest = 60
HR_max = 200

# --------------------------------
# 🧮 Feature Engineering
# --------------------------------
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# TRIMP calculation
data['HR_reserve_ratio'] = (data['HR_avg'] - HR_rest) / (HR_max - HR_rest)
data['TRIMP'] = data['Duration'] * data['HR_reserve_ratio'] * np.exp(1.92 * data['HR_reserve_ratio'])

# EWMA ACWR (7-day acute, 28-day chronic)
data['AcuteLoad'] = data['TRIMP'].ewm(span=7, adjust=False).mean()
data['ChronicLoad'] = data['TRIMP'].ewm(span=28, adjust=False).mean()
data['ACWR_EMWA'] = data['AcuteLoad'] / data['ChronicLoad']

# Monotony
def calc_monotony(index):
    if index < 6:
        return np.nan
    week = data.iloc[index-6:index+1]['TRIMP']
    return week.mean() / week.std() if week.std() != 0 else 0

data['Monotony'] = [calc_monotony(i) for i in range(len(data))]

# Strain
data['WeeklyLoad'] = data['TRIMP'].rolling(window=7).sum()
data['Strain'] = data['WeeklyLoad'] * data['Monotony']

# Training status
def training_status(acwr):
    if acwr > 1.5:
        return "Overtraining"
    elif acwr < 0.8:
        return "Undertraining"
    else:
        return "Optimal"

data['TrainingStatus'] = data['ACWR_EMWA'].apply(training_status)

# Drop NaNs
df = data.dropna(subset=['ACWR_EMWA', 'TRIMP', 'Monotony', 'Strain', 'Injured'])

# --------------------------------
# 🧠 Machine Learning
# --------------------------------
features = ['ACWR_EMWA', 'TRIMP', 'Monotony', 'Strain']
X = df[features]
y = df['Injured']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# --------------------------------
# 📈 Plot Injury Risk Over Time
# --------------------------------
# 📈 Get injury probabilities for all records
df['InjuryRiskProb'] = model.predict_proba(df[features])[:, 1] * 100  # Convert to %

# 🗓️ Sort by date
df = df.sort_values('Date')

# 🖼️ Plot injury risk
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['InjuryRiskProb'], marker='o', label='Injury Risk (%)', color='crimson')
plt.axhline(30, color='green', linestyle='--', label='Low Risk Threshold (30%)')
plt.axhline(70, color='orange', linestyle='--', label='High Risk Threshold (70%)')
plt.fill_between(df['Date'], 0, 30, color='green', alpha=0.1)
plt.fill_between(df['Date'], 30, 70, color='yellow', alpha=0.1)
plt.fill_between(df['Date'], 70, 100, color='red', alpha=0.1)

plt.title('📉 Predicted Injury Risk Over Time')
plt.xlabel('Date')
plt.ylabel('Injury Risk (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------
# 📅 Latest Player Information
# --------------------------------
latest = df.iloc[-1]
print(f"\n📅 Date: {latest['Date'].date()}")
print(f"🔥 ACWR: {latest['ACWR_EMWA']:.2f}")
print(f"📏 TRIMP: {latest['TRIMP']:.1f}")
print(f"📊 Monotony: {latest['Monotony']:.2f}")
print(f"💥 Strain: {latest['Strain']:.1f}")
print(f"✅ Training Status: {latest['TrainingStatus']}")

# 📌 Predict injury risk as a percentage with risk levels
injury_prob = model.predict_proba([latest[features]])[0][1]  # Probability of injury
risk_percentage = injury_prob * 100

# 📊 Risk categorization
if risk_percentage < 30:
    risk_level = "Low"
elif 30 <= risk_percentage <= 70:
    risk_level = "Moderate"
else:
    risk_level = "High"

# 🖨️ Output
print(f"🚨 Predicted Injury Risk: {risk_level} ({risk_percentage:.1f}%)")