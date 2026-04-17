# Step 1: Import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 2: Load data
df = pd.read_csv('data/heart.csv')

# Step 3: Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Step 4: EDA (Day 1 work)
print(df.head())
print(df.info())
print(df.describe())

sns.countplot(x='target', data=df)
plt.show()

sns.histplot(df['age'], bins=20, kde=True)
plt.show()

sns.boxplot(x='target', y='age', data=df)
plt.show()

sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.show()

