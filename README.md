

---

# 🧠 Customer Churn Prediction using ANN

This project uses an **Artificial Neural Network (ANN)** to predict **customer churn** for a subscription-based business. It includes preprocessing, training, and deploying the model using **Streamlit** for interactive prediction.

---

## 📁 Project Structure

```
ANN Project/
│
├── logs/                             # (Optional) TensorBoard logs
├── app.py                            # Streamlit app for live predictions
├── Churn_Modelling.csv               # Dataset used for training
├── experiments.ipynb                 # Data exploration and model training
├── label_encoder_gender.pkl          # Saved LabelEncoder for gender
├── model.h5                          # Trained ANN model
├── onehot_encoder_geography.pkl      # Saved OneHotEncoder for geography
├── prediction.ipynb                  # Notebook to test predictions manually
├── scaler.pkl                        # StandardScaler used in preprocessing
└── README.md                         # Project documentation
```

---

## 💡 Problem Statement

Customer churn causes significant revenue loss. The goal of this project is to **predict whether a customer will leave or stay**, based on their profile and behavior.

---

## 📊 Dataset Overview

- **File**: `Churn_Modelling.csv`
- **Target variable**: `Exited` (1 = churned, 0 = retained)
- **Features**:
  - Credit Score, Geography, Gender, Age, Tenure
  - Balance, Number of Products, Has Credit Card
  - Is Active Member, Estimated Salary

---

## 🧰 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pickle (for saving encoders and scaler)

---

## ⚙️ Model Overview

- **Model Type**: Artificial Neural Network using Keras
- **Architecture**: Sequential Model
- **Activation Function**: ReLU (hidden), Sigmoid (output)
- **Loss Function**: Binary Crossentropy
- **Saved Model**: `model.h5`

---

## 📈 Model Performance

| Metric     | Value     |
|------------|-----------|
| **Accuracy**   | 86.25%    |
| **Precision**  | 72.52%    |
| **F1 Score**   | 58.02%    |

> 🔎 Note: F1 Score is lower than Accuracy due to potential class imbalance or fewer correctly predicted churn cases.

---

## 🧪 How to Evaluate the Model

```python
from sklearn.metrics import accuracy_score, precision_score, f1_score

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
```

---

## 🚀 How to Run the Streamlit App

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ 2. Launch the App

```bash
streamlit run app.py
```

### 🖥️ 3. Use the Interface

- Enter customer details via UI
- Click Predict → Get result: **Churned or Not**

---

## 🔮 Possible Improvements

- Handle class imbalance (e.g., SMOTE or class weights)
- Hyperparameter tuning (Dropout, Optimizer, Layers)
- Add SHAP or LIME for interpretability
- Better UI with confidence scores

---

## 👤 Author

Moulik Zinzala  
Aspiring Data Scientist  
📧 moulikzinzala912@gmail.com 
🔗 [GitHub](https://github.com/Moulik-23) | [LinkedIn](www.linkedin.com/in/moulik-zinzala-2749752b7)

---

