# IRIS FLOWER CLASSIFICATION (SAFE VERSION)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Iris.csv")

print("Columns in dataset:")
print(df.columns)

# Automatically detect target column
target_column = df.columns[-1]   # Last column is usually species

# Features & Target
X = df.iloc[:, 1:5] if "Id" in df.columns else df.iloc[:, 0:4]
y = df[target_column]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Iris Model Accuracy:", accuracy_score(y_test, y_pred))