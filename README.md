# 🚗 BMW Car Price Prediction – End-to-End ML Project

## 📌 Project Overview

This project is an **end-to-end Machine Learning application** that predicts the **price of used BMW cars** based on historical data and inputs.
It covers the **complete ML lifecycle** — from data ingestion to model deployment — and exposes predictions through a **FastAPI-powered web UI**.

---

## 🎯 Key Objectives

* Design a **modular ML pipeline**
* Handle **real-world data issues in transformation stage** (like missing values, trailing spaces, unseen labels)
* Ensure **feature consistency** between training and inference
* Serve predictions using **FastAPI + HTML**
* Improve prediction reliability using **confidence intervals**

---

## 🛠️ Tech Stack

### Backend & ML

* Python 3.8
* FastAPI
* Pandas, NumPy
* Scikit-learn
* Joblib

### Frontend

* HTML (Jinja2 templates)

### Experimentation

* Jupyter Notebooks

---

## 📊 Dataset

Dataset Source:
👉 https://www.kaggle.com/datasets/algozee/bmw-dataset/data


The dataset contains historical BMW car listings with features such as model, year, mileage, fuel type, transmission, and price.

---

## 📂 Project Structure

```
├── app.py                      # FastAPI app (UI + prediction endpoint)
├── main.py                     # Pipeline execution entry
├── artifacts/                  # All generated artifacts
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   │   ├── feature_names.pkl
│   │   ├── label_encoders.pkl
│   │   ├── scaler.pkl
│   │   ├── train.csv
│   │   └── test.csv
│   ├── model_trainer/
│   │   └── model.pkl
│   └── model_evaluation/
│       └── metrics.json
│
├── config/
│   └── config.yaml             # Pipeline configuration
├── params.yaml                 # Model parameters
├── schema.yaml                 # Data schema
├── logs/
│   └── running_log.log
│
├── research/                   # EDA & experiments
│   ├── notebooks              
│   └── plots
│
├── src/
│   └── car_price/
│       ├── components/         # Core ML logic
│       ├── pipeline/           # Pipeline stages
│       ├── config/             # Configuration manager
│       ├── entity/             # Config entities
│       └── utils/              # Common utilities
│
├── templates/
│   └── index.html              # Prediction UI
├── requirements.txt
├── setup.py
├── README.md
```

---

## 🔁 ML Pipeline Stages

1. **Data Ingestion**

   * Load and store BMW dataset
2. **Data Validation**

   * Schema checks
   * Validation status logging
3. **Data Transformation**

   * Feature engineering (`car_age`)
   * Label encoding
   * Feature scaling
   * Saving:

     * `label_encoders.pkl`
     * `scaler.pkl`
     * `feature_names.pkl`
4. **Model Training**

   * Random Forest Regression model training
   * Model persistence
5. **Model Evaluation**

   * Performance metrics saved as JSON
6. **Prediction Pipeline**

   * Handles real-time user inputs
   * Ensures feature order & encoding consistency

---

## 📊 Input Features

* Model - BMW vehicle model
* Year - Year of manufacture
* Transmission - Type of transmission
* Mileage - Total distance driven as displayed in odometer reading.
* Fuel Type - Type of fuel used(i.e, power support for engine)
* Road Tax - Road tax amount
* MPG - Fuel efficiency (miles/gallon
* Engine Size - Engine size in liters

---

## 📈 Output

* **Predicted Car Price**
* **Confidence Interval**

  * Lower bound
  * Upper bound

---

## 📊 Dataset

* **Dataset Source:**
  👉 *kaggle.com(by Muhammad Shahzad)*

  ```
  [https://www.kaggle.com/datasets/algozee/bmw-dataset/data]
  ```

* The dataset contains historical BMW car listings with features such as model, year, mileage, fuel type, transmission, and price.

---

## 💱 Currency Assumption

* The original dataset **did not explicitly specify the currency** for car price and road tax.
* **British Pounds (£)** as the default currency for:

  * Model training
  * Predictions
  * UI display

---

## 🚀 How to Run the Project Locally

Follow the steps below to set up and run the project on your local machine.

### 1️⃣ Create and Activate Conda Environment

```bash
conda create -n 'env_name' python=version name -y
conda activate 'env_name'
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the End-to-End ML Pipeline

This will execute all pipeline stages (data ingestion → validation → transformation → training → evaluation) and generate artifacts and logs.

```bash
python main.py
```

📌 Pipeline logs will be available in the `logs/` directory.

---

### 4️⃣ Test Model Prediction (CLI Test)

Run a quick sanity check to ensure the trained model is working correctly:

```bash
python test_prediction.py
```

---

### 5️⃣ Run the Web Application (FastAPI + HTML UI)

Start the FastAPI server to access the interactive UI:

```bash
uvicorn app:app --reload
```

---

## 🧠 Why `feature_names.pkl`?

* Ensures **feature order consistency**
* Prevents:

  * `Feature names mismatch`
  * `Unseen labels` errors
* Makes inference **robust & production-safe**

---

## 🧪 Model Evaluation / Results

The final model was evaluated on a held-out test dataset using multiple regression performance metrics to ensure robustness and generalization.

### 📊 Evaluation Metrics

| Metric       | Value       | Description                                |
| ------------ | ----------- | ------------------------------------------ |
| **R² Score** | **0.945**   | Explains ~94.5% of variance in car prices  |
| **RMSE**     | **2666.67** | Average prediction error in GBP            |
| **MAE**      | **1582.31** | Mean absolute deviation from actual prices |
| **MAPE**     | **7.35%**   | Average percentage error                   |

✅ These results indicate **strong predictive performance**.
---

### ⚙️ Selected Model & Hyperparameters

The best-performing model was **Random Forest Regressor**, selected after experimentation and evaluation.

```json
{
  "n_estimators": 300,
  "max_depth": 30,
  "min_samples_split": 5,
  "min_samples_leaf": 1,
  "max_features": "sqrt",
  "random_state": 42
}
```

🔹 The model effectively captures non-linear relationships between vehicle attributes and price.

---

## 🖥️ Web Application (UI Preview)

An interactive web application was built using **FastAPI** and **HTML**, allowing users to predict BMW car prices in real time.

### 🚗 Application Features

1. User enters car details (model, year, mileage, fuel type, transmission, road tax, mpg, engine size)
2. Inputs are preprocessed using saved encoders and scalers
3. Model generates a price prediction
4. Prediction and confidence range are displayed on the UI


---

## 🔍 Key Learnings

* End-to-end ML pipeline design
* Feature consistency between training & inference
* Handling unseen categorical values
* Deploying ML models with FastAPI
* Building user-friendly prediction UIs

---

## 🔮 Future Enhancements

* Advanced models (XGBoost, LightGBM)
* Better UI (Bootstrap / Tailwind)
* API-only inference mode
* Cloud deployment
* Model monitoring

---

## 📌 Motivation

This project was built as a **learning-focused initiative** to gain hands-on experience with **real-world ML system design and deployment challenges**.

---

## ⭐ If you like this project

Please consider **starring the repository** and sharing feedback!.
