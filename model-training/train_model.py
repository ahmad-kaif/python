# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris
data = load_iris()

# Convert it into a DataFrame for easier manipulation
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

# View the first few rows of the dataset
print(df.head())


# Separate features and target variable
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verify the scaled features
print("Scaled features:\n", X_scaled[:5])


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verify the split
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Print the trained model
print("Model trained successfully!")

