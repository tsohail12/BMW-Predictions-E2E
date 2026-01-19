# 🚗 BMW Car Price Prediction – End-to-End ML Project

## 📌 Project Overview

This project is an **end-to-end Machine Learning application** that predicts the **price of used BMW cars** based on historical data and user-provided inputs.
It covers the **complete ML lifecycle** — from data ingestion to model deployment — and exposes predictions through a **FastAPI-powered web UI**.

---

## 🎯 Key Objectives

* Design a **modular ML pipeline**
* Handle **real-world data issues** (missing values, trailing spaces, unseen labels)
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

Absolutely 👍
Here are **clean, professional additions** you can directly paste into your existing README.

---

## 📊 Dataset

* **Dataset Source:**
  👉 *Paste dataset link here*

  ```
  [ADD_DATASET_LINK_HERE]
  ```

* The dataset contains historical BMW car listings with features such as model, year, mileage, fuel type, transmission, and price.

---

## 💱 Currency Assumption

* The original dataset **did not explicitly specify the currency** for car prices.
* Assumed:
* **British Pounds (£)** as the default currency for:

  * Model training
  * Predictions
  * UI display

---

## 🧠 Why `feature_names.pkl`?

* Ensures **feature order consistency**
* Prevents:

  * `Feature names mismatch`
  * `Unseen labels` errors
* Makes inference **robust & production-safe**

---

## 🧪 Model Confidence Interval

The confidence interval provides:

* A **price range** instead of a single value
* Better **user trust**
* Practical insight into **prediction uncertainty**

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
