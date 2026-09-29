import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

csv_filename = "student_data.csv"
if not os.path.exists(csv_filename):
    print(f"'{csv_filename}' not found. Generating a 100-row dataset automatically...")
  
    with open(csv_filename, "w") as f:
        f.write(sample_data)

df = pd.read_csv(csv_filename)


X = df[['Attendance', 'Study hours', 'internal_marks', 'participation']]
y = df['result']

model = RandomForestClassifier(n_estimators=3, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

print("Total samples trained on:", len(df))
print("\nFirst 10 Actual results:   ", list(y[:10]))
print("First 10 Predicted results:", list(y_pred[:10]))

accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy on training data: {accuracy * 100:.2f}%")

new_student = pd.DataFrame([[30, 2, 30, 3]], columns=X.columns)
prediction = model.predict(new_student)
print("New Student prediction:", prediction[0])

plt.bar(X.columns, model.feature_importances_) 
plt.xlabel("Features")                         
plt.ylabel("Importance Score")
plt.title("Random Forest Feature Importance (100 Rows)")
plt.xticks(rotation=20)                        
plt.tight_layout()                              
plt.show()
