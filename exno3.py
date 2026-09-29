import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = {
    'Age': [23, 32, 42, 47, 60],
    'Weight': [55, 68, 75, 84, 90],          
    'Blood_Pressure': [100, 110, 117, 120, 135],
    'Medical_Condition': [0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)
print("Dataset")
print(df)
X = df[['Age', 'Weight', 'Blood_Pressure']]
y = df['Medical_Condition']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nActual Output")
print(y_test.values)
print("\nPredicted Output")
print(y_pred)
print("\nIntercept")
print(model.intercept_)
print("\nCoefficient")
print(model.coef_)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy")
print(accuracy * 100, "%")
age = pd.DataFrame({'Age': [40], 'Weight': [75], 'Blood_Pressure': [128]})
print("\nProbability")
print(model.predict_proba(age))
x = np.linspace(20,65,100).reshape(-1,1)
x_plot = pd.DataFrame({
    'Age': x.flatten(),
    'Weight': np.full(100, df['Weight'].mean()),
    'Blood_Pressure': np.full(100, df['Blood_Pressure'].mean())
})
y_prob = model.predict_proba(x_plot)[:,1]
plt.scatter(df['Age'],df['Medical_Condition'])
plt.plot(x,y_prob)
plt.xlabel("Age")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.grid(True)
plt.show()
