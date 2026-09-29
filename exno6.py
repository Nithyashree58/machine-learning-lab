import matplotlib.pyplot as plt
data = [
    (20, 6, 16, "Yes"),
    (30, 12, 25, "Yes"),
    (35, 13, 25, "No"),
    (46, 15, 30, "No"),
    (36, 10, 24, "Yes"),
    (38, 13, 26, "Yes"),
    (33, 10, 28, "Yes"),
    (27, 6, 24, "Yes"),
    (22, 6, 12, "No"),
    (25, 8, 15, "No"),
    (35, 10, 20, "No"),
    (48, 14, 11, "No"),
    (40, 7, 18, "No"),
    (32, 5, 27, "No")
]
def age_cat(v):
    return "Middle" if 30 <= v <= 40 else "Other"


def brow_cat(v):
    return "High" if v > 10 else "Low"


def time_cat(v):
    return "Long" if v > 20 else "Short"
a = age_cat(33)
b = brow_cat(13)
t = time_cat(27)

print("New User Profile:")
print(f"Age = {a}")
print(f"Browsing = {b}")
print(f"Time Spent = {t}\n")
yes = [x for x in data if x[3] == "Yes"]
no = [x for x in data if x[3] == "No"]

print("Number of Yes:", len(yes))
print("Number of No :", len(no))
print("Total data   :", len(data))
print()

p_yes = len(yes) / len(data)
p_no = len(no) / len(data)

ay = sum(age_cat(x[0]) == a for x in yes) / len(yes)
by = sum(brow_cat(x[1]) == b for x in yes) / len(yes)
ty = sum(time_cat(x[2]) == t for x in yes) / len(yes)
an = sum(age_cat(x[0]) == a for x in no) / len(no)
bn = sum(brow_cat(x[1]) == b for x in no) / len(no)
tn = sum(time_cat(x[2]) == t for x in no) / len(no)
py = ay * by * ty * p_yes
pn = an * bn * tn * p_no

print("Naive Bayes Probabilities:")
print(f"P(Yes) = {py:.4f}")
print(f"P(No)  = {pn:.4f}")
if py > pn:
    prediction = "Yes"
else:
    prediction = "No"

print(f"\nFinal Prediction: Click = {prediction}")

labels = ["Click = Yes", "Click = No"]
probabilities = [py, pn]
colors = ["#2ecc71", "#e74c3c"]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    labels,
    probabilities,
    color=colors,
    width=0.5
)
for bar, value in zip(bars, probabilities):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=11
    )


plt.xlabel("Prediction")
plt.ylabel("Probability")
plt.title("Naive Bayes Prediction Comparison")

plt.ylim(0, max(probabilities) * 1.3)

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()
