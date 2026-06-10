# Iris-Flower-Classification-using-decision-trees

### 📋 Overview

This project implements a classic Machine Learning classification pipeline to identify Iris flower species (**Setosa, Versicolor, and Virginica**) based on their morphological measurements. By utilizing a **Decision Tree Classifier**, the system builds a transparent, rule-based model that mimics human decision-making.

### 🧠 How the Model Thinks

A Decision Tree splits the dataset into smaller subsets based on the features that provide the highest **Information Gain** (or lowest **Gini Impurity**).

For the Iris dataset, the tree typically discovers that:

* If **Petal Length** is very short -> It is instantly classified as **Setosa**.
* If **Petal Length** is longer, it checks **Petal Width** to separate **Versicolor** from **Virginica**.

---

### 🚀 Key Features

* **White-Box Modeling**: The resulting model is fully interpretable, allowing you to trace exactly *why* a flower was classified a certain way.
* **No Feature Scaling Required**: Decision Trees are scale-invariant, meaning they perform perfectly without normalizing or standardizing the data.
* **Tree Visualization**: Generates a visual flowchart of the trained decision tree to analyze the exact splitting thresholds.

---

### 🛠️ Tech Stack

* **Language**: Python 3.x
* **Machine Learning**: `scikit-learn`
* **Data Handling**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`

---

### 📂 Dataset Features

The model trains on the famous **Fisher's Iris Dataset**, which contains 150 samples with 4 physical attributes:

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

---

### 📊 Model Evaluation

Because the Iris dataset is balanced (50 samples per class), **Accuracy** is a reliable metric for this project.

| Species | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **Setosa** | 1.00 | 1.00 | 1.00 |
| **Versicolor** | 0.94 | 0.94 | 0.94 |
| **Virginica** | 0.94 | 0.94 | 0.94 |

---

### 📦 Quick Start

1. **Install Dependencies**:
```bash
pip install scikit-learn pandas matplotlib seaborn

```


2. **Train and Visualize**:
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Initialize and fit
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X, y)

# Plot the tree structural rules
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

```



---

### 🛡️ Preventing Overfitting

To ensure the tree doesn't simply "memorize" the training data, this project uses hyperparameter tuning parameters such as:

* `max_depth`: Restricting how deep the tree can grow.
* `min_samples_split`: The minimum number of samples required to split an internal node.
