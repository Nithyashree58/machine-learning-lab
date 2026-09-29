import math
import matplotlib.pyplot as plt
data = [
    (27, 120, 185, "no"),
    (31, 125, 195, "no"),
    (29, 116, 175, "no"),
    (49, 148, 238, "yes"),
    (54, 142, 255, "yes"),
    (51, 136, 268, "yes")
]
new_patient = (34, 132, 205)
distances = []
for age, bp, chol, disease in data:
    d = math.sqrt(
        (age - new_patient[0]) ** 2 +
        (bp - new_patient[1]) ** 2 +
        (chol - new_patient[2]) ** 2
    )
    distances.append((d, disease, age, bp, chol))
distances.sort()
print("Distances:")
for d, disease, age, bp, chol in distances:
    print("Distance:", round(d, 2), "Disease:", disease)
k = 3
nearest = distances[:k]
print("\nNearest 3 neighbours:")
for d, disease, age, bp, chol in nearest:
    print(
        "Age:", age,
        "BP:", bp,
        "Chol:", chol,
        "Distance:", round(d, 2),
        "Disease:", disease
    )
yes_count = sum(1 for d, disease, age, bp, chol in nearest
                if disease == "yes")
no_count = sum(1 for d, disease, age, bp, chol in nearest
               if disease == "no")
if yes_count > no_count:
    prediction = "yes"
else:
    prediction = "no"
print("\nYes count:", yes_count)
print("No count:", no_count)
print("Final prediction:", prediction)
yes_patients = [row for row in data if row[3] == "yes"]
no_patients = [row for row in data if row[3] == "no"]
plt.figure(figsize=(8, 6))
plt.scatter(
    [row[0] for row in no_patients],
    [row[2] for row in no_patients],
    color="blue",
    label="Disease = No",
    s=100
)
plt.scatter(
    [row[0] for row in yes_patients],
    [row[2] for row in yes_patients],
    color="red",
    label="Disease = Yes",
    s=100
)
plt.scatter(
    new_patient[0],
    new_patient[2],
    color="green",
    marker="*",
    s=300,
    label="New Patient"
)
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("KNN Classification - Age vs Cholesterol")
plt.legend()
plt.grid(True)
plt.show()
