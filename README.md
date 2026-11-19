# Iris Flower Classification using Decision Trees

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3776AB?logo=python&logoColor=white)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An end-to-end Machine Learning pipeline that classifies Iris flower species (**Setosa**, **Versicolor**, and **Virginica**) using physical flower measurements and a **Decision Tree Classifier** from `scikit-learn`.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Summary](#-dataset-summary)
- [Used Libraries](#-used-libraries)
- [Project Workflow](#-project-workflow)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Decision Tree Visualization](#-decision-tree-visualization)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🌸 Overview

The project uses morphological measurements of Iris flowers to build a transparent, rule-based classification model. A **Decision Tree Classifier** recursively splits the data by selecting feature thresholds that minimize node impurity (Gini Impurity), producing an intuitive and fully explainable decision model.

---

## 📂 Dataset Summary

The project utilizes Fisher's classic **Iris Dataset** loaded directly via `sklearn.datasets.load_iris`:

* **Total Samples**: 150 instances (50 samples per class)
* **Target Classes**:
  * `0`: *Iris Setosa*
  * `1`: *Iris Versicolor*
  * `2`: *Iris Virginica*
* **Input Features** (4 numerical attributes):
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`

---

## 🛠️ Used Libraries

The implementation in [`program.ipynb`](program.ipynb) relies on the following standard libraries:

| Library | Modules / Functions Used | Purpose |
| :--- | :--- | :--- |
| **`pandas`** | `pd.DataFrame` | Structuring feature arrays and target labels into tabular format |
| **`scikit-learn`** | `load_iris` | Loading the pre-packaged Iris dataset |
| | `train_test_split` | Splitting data into training (70%) and testing (30%) subsets |
| | `DecisionTreeClassifier` | Initializing and training the Decision Tree model |
| | `metrics` (`accuracy_score`, `confusion_matrix`, `classification_report`) | Model evaluation and performance metrics |
| | `plot_tree` | Rendering the visual tree architecture |
| **`matplotlib`** | `matplotlib.pyplot` | Setting up figure dimensions and displaying the tree diagram |

---

## ⚙️ Project Workflow

The notebook [`program.ipynb`](program.ipynb) executes the following sequential steps:

1. **Import Libraries**: Load data handling, modeling, metric, and visualization tools.
2. **Load Dataset**: Fetch Iris data and wrap it into a `pandas.DataFrame`.
3. **Feature-Target Separation**:
   * Features ($X$): `sepal length`, `sepal width`, `petal length`, `petal width`
   * Target ($y$): `species`
4. **Train-Test Split**:
   * 70% Training data (105 samples)
   * 30% Testing data (45 samples)
   * Set `random_state=42` for reproducibility.
5. **Model Training**:
   * Train a `DecisionTreeClassifier()` on `X_train` and `y_train`.
6. **Inference & Prediction**:
   * Predict classes for the unseen test set `X_test`.
7. **Performance Evaluation**:
   * Compute Accuracy, Confusion Matrix, and Classification Report.
8. **Decision Tree Visualization**:
   * Generate and display the decision flowchart using `plot_tree`.

---

## 📊 Model Evaluation & Results

Evaluated on the test split ($n = 45$ samples):

### 1. Overall Accuracy
$$\text{Accuracy} = \mathbf{1.0\ (100\%)}$$

> **Note on Perfect Accuracy / 0 Error**: With `test_size=0.3` and `random_state=42`, all 45 test instances fall cleanly within the decision tree's partition boundaries. While 100% accuracy is achieved here on this clean benchmark dataset, testing across different seeds or cross-validation typically yields ~93%–97% generalization accuracy.

### 2. Confusion Matrix
```text
[[19  0  0]   <- Setosa (19/19 correct)
 [ 0 13  0]   <- Versicolor (13/13 correct)
 [ 0  0 13]]  <- Virginica (13/13 correct)
```

### 3. Classification Report
| Class | Species | Precision | Recall | F1-Score | Support |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | *Setosa* | 1.00 | 1.00 | 1.00 | 19 |
| **1** | *Versicolor* | 1.00 | 1.00 | 1.00 | 13 |
| **2** | *Virginica* | 1.00 | 1.00 | 1.00 | 13 |
| **Total / Avg** | **Accuracy** | | | **1.00** | **45** |
| | **Macro Avg** | 1.00 | 1.00 | 1.00 | 45 |
| | **Weighted Avg**| 1.00 | 1.00 | 1.00 | 45 |

---

## 🌳 Decision Tree Visualization

The decision tree generated in the notebook visually exposes the decision boundaries:

```python
plt.figure(figsize=(12, 8))
plot_tree(classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Desion tree for Iris flower Classification")
plt.show()
```

* **First Split**: Petal measurements cleanly isolate *Setosa* with a single condition (`petal length <= 2.45 cm`).
* **Subsequent Splits**: Deeper nodes evaluate `petal width` and `petal length` to differentiate between *Versicolor* and *Virginica*.
* **Leaf Nodes**: Every leaf achieves Gini impurity = `0.0`, indicating complete class purity.

---

## 📁 Project Structure

```text
Iris-Flower-Classification-using-decision-trees/
├── .gitignore          # Git ignore specifications
├── LICENSE             # Apache 2.0 License
├── README.md           # Project documentation and guide
└── program.ipynb       # Jupyter Notebook with code, outputs, and visualization
```

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed on your machine.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Iris-Flower-Classification-using-decision-trees.git
   cd Iris-Flower-Classification-using-decision-trees
   ```

2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib notebook
   ```

### Running the Notebook

Launch Jupyter Notebook or VS Code to run the analysis:

```bash
jupyter notebook program.ipynb
```

Execute the cells sequentially to reproduce data exploration, model training, metric computation, and tree visualization.

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
