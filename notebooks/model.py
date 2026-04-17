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

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))