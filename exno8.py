import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC

data = {

"BloodPressure": [110, 115, 120, 130, 140, 150],

"BMI": [19, 21, 22, 26, 29, 32],

"Condition": [0, 0, 0, 1, 1, 1]

}

df = pd.DataFrame(data)

X = df[["BloodPressure", "BMI"]]

y = df["Condition"]

model = SVC(kernel="linear", C=1)

model.fit(X, y)

blood_pressure = 140

bmi = 29

new_patient = pd.DataFrame(

[[blood_pressure, bmi]],

columns=["BloodPressure", "BMI"]

)

prediction = model.predict(new_patient)

print("New Patient")

print("Blood Pressure:", blood_pressure)

print("BMI:", bmi)

if prediction[0] == 0:
    print("Predicted Condition: No Condition")

else:
    print("Predicted Condition: Condition")
    print("Accuracy:", model.score(X, y) * 100, "%")
    print("Weights:", model.coef_[0])
    print("Bias:", model.intercept_[0])

plt.scatter(

df[df.Condition == 0]["BloodPressure"],

df[df.Condition == 0]["BMI"],

label="No Condition"

)

plt.scatter(

df[df.Condition == 1]["BloodPressure"],

df[df.Condition == 1]["BMI"],

label="Condition"

)

w = model.coef_[0]

b = model.intercept_[0]

blood_pressure_values = [110, 150]

bmi_values = [

-(w[0] * bp + b) / w[1]

for bp in blood_pressure_values

]

plt.plot(

blood_pressure_values,

bmi_values,

label="Decision Boundary"

)

plt.scatter(

blood_pressure,

bmi,

marker="*",

s=150,

label="New Patient"

)

plt.xlabel("Blood Pressure")

plt.ylabel("BMI")

plt.title("SVM Classification")

plt.legend()

plt.grid()

plt.show()
