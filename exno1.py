import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

temperature = np.array([5, 10, 15, 21, 27]).reshape(-1, 1)
sales = np.array([50, 70, 90, 110, 130])
model = LinearRegression()
model.fit(temperature, sales)

print("Intercept (b):", model.intercept_)
print("Slope (m):", model.coef_[0])

temp = np.array([[35]])
predicted_sales = model.predict(temp)

print("Predicted Ice Cream Sales at 35°C:", predicted_sales[0])


plt.scatter(temperature, sales, color='blue', label='Data Points')
plt.plot(temperature, model.predict(temperature), color='red', label='Regression Line')
plt.scatter(35, predicted_sales, color='green', s=100, label='Prediction (35°C)')
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
