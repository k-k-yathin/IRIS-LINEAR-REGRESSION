import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("IRIS_dataset.csv")
print("First 5 rows of dataset are:")
print(df.head())

X = df[['sepal_length', 'sepal_width', 'petal_width']]
y = df['petal_length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

print("\nWeights (coefficients):", model.coef_)
print("Bias (intercept):", model.intercept_)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Actual vs Predicted Values")
plt.show()