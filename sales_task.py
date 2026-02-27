# =====================================
# SALES PREDICTION USING LINEAR REGRESSION
# =====================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# 1️⃣ Load Dataset
df = pd.read_csv("advertising.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 2️⃣ Data Visualization
sns.pairplot(df)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 3️⃣ Features & Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5️⃣ Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Prediction
y_pred = model.predict(X_test)

# 7️⃣ Evaluation
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 8️⃣ Example Prediction
sample = [[150, 25, 10]]  # TV, Radio, Newspaper
predicted_sales = model.predict(sample)
print("\nPredicted Sales:", predicted_sales[0])