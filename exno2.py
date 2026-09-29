import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = {
    "x1": [5, 6, 7],
    "x2": [900, 1250, 1500],
    "y": [39, 56, 67]
}
df = pd.DataFrame(data)
x = df[["x1", "x2"]]
y = df["y"]
model = LinearRegression()
model.fit(x, y)
print("Intercept:", model.intercept_)
print("Coefficient X1:", model.coef_[0])
print("Coefficient X2:", model.coef_[1])
new = pd.DataFrame({"x1": [2], "x2": [180]})
new_prediction = model.predict(new)
print("Prediction for new data:", new_prediction[0])
prediction = model.predict(x)
plt.figure(figsize=(6,4))
plt.plot(y.values, marker='o', label='Actual')
plt.plot(prediction, marker='s', label='Predicted')
plt.title("Actual vs Predicted")
plt.xlabel("Observation")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
