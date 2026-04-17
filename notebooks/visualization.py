import pandas as pd

# Load dataset
df = pd.read_csv('data/heart.csv')

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# Features & target
X = df.drop('target', axis=1)
y = df['target']

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import matplotlib.pyplot as plt
import numpy as np

importances = model.feature_importances_
features = X.columns

# Sort features
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.tight_layout()
plt.show()

print("\nTop Important Features:")
for i in indices[:5]:
    print(f"{features[i]}: {importances[i]:.4f}")

import seaborn as sns

plt.figure(figsize=(8,6))
sns.boxplot(x='target', y='Max_heart_rate', data=df)
plt.title("Max Heart Rate vs Disease")
plt.show()