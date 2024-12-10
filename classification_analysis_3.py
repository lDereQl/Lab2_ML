# Re-import necessary libraries due to state reset
import matplotlib.pyplot as plt
import numpy as np

# Metrics for both models
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
custom_scores = [0.74, 0.74, 0.74, 0.74]
sklearn_scores = [0.75, 0.75, 0.75, 0.75]

# Bar chart
x = np.arange(len(metrics))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, custom_scores, width, label="Custom Model")
bars2 = ax.bar(x + width / 2, sklearn_scores, width, label="Scikit-learn Model")

# Labels and title
ax.set_xlabel("Metrics")
ax.set_ylabel("Scores")
ax.set_title("Comparison of Custom and Scikit-learn Decision Tree Models")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()


# Adding data labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset label above the bar
                    textcoords="offset points",
                    ha='center', va='bottom')


add_labels(bars1)
add_labels(bars2)

# Display plot
plt.tight_layout()
plt.show()
