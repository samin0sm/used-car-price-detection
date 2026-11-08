# 🚗 Used Car Price Detection

A machine learning web application built with Streamlit to predict the market price of used cars based on mileage and model year.

---

## 📌 Project Overview

This project uses a Linear Regression model trained on car sales data (analyzed in `notebooks/Regression_of_Used_Car_Prices.ipynb`). Through Recursive Feature Elimination (RFE), the most predictive features identified were:
- **`milage`**: Total miles driven
- **`model_year`**: Manufacturing year of the car

---

## 📁 Project Structure

```text
used car price detection/
│
├── dataset/
│   └── train.csv               # Dataset used for training
├── model/
│   ├── model.pkl               # Trained Linear Regression model
│   └── scaler.pkl              # Feature scaler
├── notebooks/
│   └── Regression_of_Used_Car_Prices.ipynb # Exploratory data analysis & model training
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.


---
