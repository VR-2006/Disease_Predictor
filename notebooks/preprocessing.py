# Step 1: Import library
import pandas as pd

# Step 2: Load dataset
df = pd.read_csv('data/heart.csv')

print("Dataset loaded successfully!")

# Step 3: Convert categorical data to numeric
df = pd.get_dummies(df, drop_first=True)

print("Categorical data converted to numeric!")

# Step 4: Split features and target
X = df.drop('target', axis=1)
y = df['target']

print("Features and target separated!")

# Step 5: Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split into training and testing!")

# Step 6: Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed!")

print("\nAll preprocessing steps completed successfully!")