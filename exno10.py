import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report 

csv_filename = "diabetes.csv"

if not os.path.exists(csv_filename):
    print(f"'{csv_filename}' not found. Generating 100 rows of dataset automatically...")
    sample_data = """Age,BMI,Bloodpressure,glucose,diabetes
59,33.2,86,188,1\n49,28.0,94,161,1\n35,21.1,60,158,0\n63,29.1,94,172,1\n28,19.2,96,131,0
41,38.1,106,166,1\n59,24.1,73,70,0\n39,32.7,62,96,0\n43,25.2,60,131,0\n31,29.7,64,185,1
31,30.3,85,146,1\n44,22.5,73,174,1\n56,39.3,98,72,0\n60,35.2,86,172,1\n44,38.7,68,139,1
25,20.5,64,88,0\n50,31.4,80,150,1\n38,26.8,72,112,0\n47,34.9,88,165,1\n62,23.3,70,95,0
33,29.2,76,140,1\n45,36.4,92,178,1\n52,22.0,62,85,0\n29,25.6,68,102,0\n55,32.1,84,159,1
40,28.7,78,124,0\n36,33.5,82,143,1\n61,30.1,90,181,1\n27,19.8,58,79,0\n43,27.4,74,115,0
58,35.6,86,168,1\n32,24.3,66,99,0\n48,37.2,96,173,1\n51,21.5,60,91,0\n34,30.8,78,135,1
46,29.4,84,142,1\n57,33.9,88,160,1\n26,22.1,62,83,0\n42,36.8,94,170,1\n53,24.8,70,105,0
37,28.3,76,121,0\n60,32.7,86,155,1\n30,26.2,68,110,0\n49,38.5,102,182,1\n54,23.1,64,89,0
39,31.2,80,138,1\n44,27.9,76,118,0\n56,34.4,90,163,1\n28,21.0,60,86,0\n41,35.9,92,166,1
52,29.8,82,145,1\n35,25.0,70,103,0\n47,33.1,86,152,1\n63,22.6,62,92,0\n31,30.5,78,139,1
45,37.7,98,176,1\n59,26.7,74,114,0\n33,23.9,66,97,0\n50,34.8,88,161,1\n38,28.5,78,126,0
46,32.4,84,149,1\n62,31.0,90,179,1\n27,20.1,60,81,0\n43,29.1,80,132,1\n58,36.2,92,171,1
32,24.9,68,100,0\n48,38.1,100,180,1\n51,22.3,64,90,0\n34,31.5,82,141,1\n46,28.9,82,123,0
57,34.2,88,162,1\n26,21.5,62,84,0\n42,37.1,96,174,1\n53,25.3,72,107,0\n37,27.8,74,119,0
60,33.0,86,157,1\n30,25.8,66,108,0\n49,38.9,104,185,1\n54,23.6,64,93,0\n39,31.7,82,142,1
44,28.2,78,120,0\n56,34.7,92,167,1\n28,21.4,62,88,0\n41,36.3,94,169,1\n52,30.1,84,148,1
35,25.4,72,106,0\n47,33.6,88,156,1\n63,22.9,64,94,0\n31,31.0,80,143,1\n45,38.2,100,179,1
59,27.1,76,116,0\n33,24.2,68,98,0\n50,35.1,90,164,1\n38,28.9,80,128,0\n46,32.8,86,151,1
62,31.4,92,182,1\n27,20.4,62,82,0\n43,29.4,82,134,1\n58,36.5,94,174,1\n32,25.2,70,102,0"""
    with open(csv_filename, "w") as f:
        f.write(sample_data)


data = pd.read_csv(csv_filename)
print(f"Dataset successfully loaded. Total records: {len(data)}")


X = data[['Age', 'BMI', 'Bloodpressure', 'glucose']]
y = data['diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)

model = AdaBoostClassifier(estimator=weak_learner,
                         n_estimators=5,
                         learning_rate=1.0,
                         random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Sample Validation Predictions ===")
print("Actual values:   ", y_test.values[:15])
print("Predicted values:", y_pred[:15])


accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy Score: {accuracy * 100:.2f}%")

print("\n=== Model Performance Evaluation ===")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(8, 5))
plt.bar(X.columns, model.feature_importances_, color='skyblue', edgecolor='black') 
plt.xlabel("Features (Health Metrics)")                         
plt.ylabel("Importance Score")
plt.title("AdaBoost Feature Importance (100 Rows)")
plt.xticks(rotation=15)                        
plt.tight_layout()                              
plt.show()
