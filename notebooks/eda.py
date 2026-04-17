# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 2: Check current working directory (for debugging)
print("Current Working Directory:", os.getcwd())

# Step 3: Load dataset
df = pd.read_csv('data/heart.csv')

# Step 4: Basic info
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATA SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns)

print("\n--- DATA INFO ---")
print(df.info())

print("\n--- STATISTICS ---")
print(df.describe())

# Step 5: Target distribution
print("\n--- TARGET DISTRIBUTION ---")
print(df['target'].value_counts())

# Step 6: Visulaization

# Target count plot
sns.countplot(x='target', data=df)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Age distribution
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()


sns.boxplot(x='target', y='age', data=df)
plt.title("Age vs Disease")
plt.show()