# End-to-End-ANN-Project

# 🏦 Customer Churn Prediction using ANN

A deep learning web application that predicts whether a bank customer will churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras and deployed with Streamlit.


## 📌 Project Overview
Customer churn is a critical problem for banks. This project builds a binary classification model to predict whether a customer will leave the bank based on their profile and account information.

---

## 🗂️ Project Structure
```
ANN-Project/
├── app.py                       # Streamlit web application
├── experiments.ipynb            # Model training notebook
├── Churn_Modelling.csv          # Dataset
├── model.h5                     # Trained ANN model
├── scaler.pkl                   # StandardScaler object
├── onehot_encode_geo.pkl        # OneHotEncoder for Geography
├── Label_encoder_gender.pkl     # LabelEncoder for Gender
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

---

## 🛠️ Tech Stack
| Tool | Purpose |
| Python 3.x | Programming Language |
| TensorFlow / Keras | ANN Model Building |
| Scikit-learn | Preprocessing & Encoding |
| Pandas & NumPy | Data Manipulation |
| Streamlit | Web Application |
| TensorBoard | Model Training Visualization |
| Pickle | Saving Encoders & Scaler |

---

## 📊 Dataset
- **Source:** Churn_Modelling.csv
- **Rows:** 10,000 customers
- **Target:** `Exited` (1 = Churned, 0 = Stayed)



## 🧠 Model Architecture
```
Input Layer  →  11 Features
Hidden Layer 1  →  64 Neurons, ReLU Activation
Hidden Layer 2  →  32 Neurons, ReLU Activation
Output Layer  →  1 Neuron, Sigmoid Activation

