# 🩺 Breast Cancer Detection System

A machine learning model that predicts whether a breast tumor is **benign** or **malignant** based on features extracted from breast mass images. Built using the **Breast Cancer Wisconsin Diagnostic Dataset** and trained with **Logistic Regression**.

---

##  Project Overview

This project aims to assist in early breast cancer detection by automating diagnosis based on patient test data. Using scikit-learn's logistic regression, the model provides a fast and reasonably accurate method of binary classification (Benign or Malignant).

---

##  Technologies Used

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| Python           | Programming Language             |
| NumPy & Pandas   | Data handling & preprocessing     |
| scikit-learn     | Model training & evaluation       |

---

##  Dataset Information

- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Total Samples**: 569
- **Features**: 30 numeric features (e.g., radius, texture, area, smoothness)
- **Target**:
  - `0` = Malignant (cancerous)
  - `1` = Benign (non-cancerous)

---

##  Data Preprocessing

- Converted `sklearn` dataset into a Pandas DataFrame.
- Added `label` column from the target.
- Checked and confirmed **no missing values** in the dataset.
- Feature and target split:
```python
x = df.drop(columns='label', axis=1)
y = df['label']
```

---

## ✂ Train-Test Split

Used an 80-20 split:
```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
```

---

##  Model Training: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(max_iter=10000)
regressor.fit(x_train, y_train)
```

_Note: Increased `max_iter` to avoid convergence warnings._

---

##  Model Evaluation

- **Training Accuracy**: 94.95%
- Evaluated using `accuracy_score`:
```python
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, regressor.predict(x_train))
```

---

##  Prediction System

Given new input features (30 values), the system predicts the cancer type:
```python
input_data = (17.99, 10.38, 122.80, ..., 0.1189)
input_np = np.asarray(input_data).reshape(1, -1)
prediction = regressor.predict(input_np)

if prediction[0] == 0:
    print("Malignant Tumor")
else:
    print("Benign Tumor")
```

---

