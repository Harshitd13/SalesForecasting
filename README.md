# 📊 Sales Forecasting & Demand Intelligence System

An end-to-end Machine Learning and Time Series Forecasting project that predicts future sales, detects anomalies, segments products based on demand patterns, and provides business insights through an interactive Streamlit dashboard.

---

## 📌 Project Overview

This project analyzes **4 years of Superstore retail sales data (2015–2018)** to help businesses improve inventory planning and demand forecasting. It combines statistical forecasting, machine learning, anomaly detection, and product segmentation into a single decision-support system.

---

## ✨ Key Features

- 📈 Exploratory Data Analysis (EDA)
- 📊 Time Series Decomposition & Seasonality Analysis
- 🤖 Sales Forecasting using:
  - SARIMA
  - Facebook Prophet
  - XGBoost
- 📦 Category-wise Sales Forecasting
- 🚨 Anomaly Detection (Isolation Forest & Z-Score)
- 🎯 Product Segmentation using K-Means Clustering
- 🌐 Interactive Streamlit Dashboard
- 📄 Executive Business Report

---

## 📂 Dataset

**Superstore Sales Dataset**

- **Transactions:** 9,800
- **Orders:** 4,922
- **Revenue:** $2.26 Million
- **Categories:** 3
- **Regions:** 4
- **Period:** 2015–2018

---

## 📊 Model Performance

| Model | MAE | RMSE | MAPE |
|------|---------:|---------:|------:|
| **SARIMA** | **19,244.49** | **19,950.07** | **20.53%** |
| Prophet | 20,296.01 | 22,487.47 | 21.89% |
| XGBoost | 28,035.82 | 28,063.84 | 31.18% |

🏆 **Best Model:** SARIMA

---

## 📈 Key Results

- 💰 Technology generated the highest revenue (**36.6% of total sales**)
- 📦 Furniture showed the strongest overall growth
- 📅 November and December consistently recorded peak sales
- 🚨 62 anomalies detected using Isolation Forest
- 🎯 17 product sub-categories segmented into demand-based clusters
- 🌐 Interactive Streamlit dashboard for business analysis

---

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Plotly, Scikit-learn, Statsmodels, Prophet, XGBoost
- **Framework:** Streamlit
- **Tools:** Jupyter Notebook, Git, GitHub

---

## 📁 Project Structure

```text
SalesForecasting_Harshit/
│
├── analysis.ipynb
├── app.py
├── dashboard_data.json
├── train.csv
├── requirements.txt
├── summary.pdf
├── README.md
└── charts/
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/Harshitd13/SalesForecasting.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Harshit Dwivedi**  
B.Tech in Computer Science & Engineering  
Jaypee University of Engineering and Technology, Guna

**GitHub:** https://github.com/Harshitd13

**LinkedIn:** https://www.linkedin.com/in/harshit-dwivedi-88472831a
---

## ⭐ Acknowledgement

This project was developed as part of an AI/ML Internship to demonstrate practical skills in time series forecasting, anomaly detection, product segmentation, and business analytics.
