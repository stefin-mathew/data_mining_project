import matplotlib.pyplot as plt

# Model names
models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM"
]

# Accuracy values
accuracy = [
    72.19,
    72.01,
    74.00,
    64.00
]

# Create graph
plt.figure(figsize=(9, 6))

bars = plt.bar(models, accuracy)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

# Display accuracy above bars
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.xticks(rotation=15)
plt.tight_layout()

plt.show()