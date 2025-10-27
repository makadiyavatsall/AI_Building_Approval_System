<h1 align="center">🏗️ AI-Based Building Approval Prediction System</h1>

<p align="center">
  <img src="https://cdn.dribbble.com/users/416610/screenshots/4801105/building-animation.gif" width="300px">
</p>



<p align="center">
  <b>Smart automation for urban infrastructure decision-making using AI & ML</b><br>
  <i>Trained to predict whether a building application will be approved or not based on multiple attributes.</i>
</p>




---

## 🧠 Project Overview
The **AI-Based Building Approval Prediction System** is a Machine Learning solution that automates the evaluation of construction proposals.  
It predicts whether a building application will be **approved or not approved**, based on various project and compliance parameters.  

This system is designed to support **urban planning departments**, **municipal authorities**, and **smart city initiatives** by reducing manual workload and improving decision-making transparency.

---

## 🎯 Objective
To develop a predictive model capable of analyzing building proposal data and automatically determining the likelihood of approval based on multiple engineered features.

---

## ⚙️ Methodology

1. **Data Collection & Cleaning** – Removed missing values, handled categorical features, and performed feature scaling.  
2. **Exploratory Data Analysis (EDA)** – Visualized relationships between project area, budget, and compliance parameters.  
3. **Feature Engineering** – Encoded categorical columns and normalized numerical variables.  
4. **Model Training** – Trained multiple ML algorithms to identify the best-performing one.  
5. **Evaluation & Optimization** – Compared models based on accuracy, precision, recall, and F1-score.  

---

## 🧩 Algorithms Used
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost / AdaBoost (Boosting Algorithms)**

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:----------:|:----------:|:-------:|:---------:|
| Logistic Regression | 82.3% | 81% | 80% | 80.5% |
| Decision Tree | 87.9% | 86% | 87% | 86.5% |
| **Random Forest** | **91.4%** | **90%** | **91%** | **90.3%** |
| Boosting (XGBoost) | 90.2% | 89% | 90% | 89.4% |

✅ **Final Model Selected:** Random Forest Classifier  
✅ **Model Saved Using:** `pickle` for deployment in the prediction pipeline.

---

## 🧪 Prediction Example

| Feature Summary | Prediction |
|-----------------|-------------|
| Plot Area: 250 sqm<br>Height: 3 Floors<br>Budget: ₹25L | ✅ Approved |
| Plot Area: 80 sqm<br>Height: 6 Floors<br>Budget: ₹15L | ❌ Not Approved |

---

## 🚀 Outcome
- Reduced manual evaluation time by **~70%**.  
- Delivered consistent and explainable predictions using machine learning.  
- Ready for integration with **Streamlit/Flask web interface** or municipal dashboards.  

---

## 👨‍💻 Author
**Vatsal Makadiya**  
B.E. Computer Engineering | Data & AI Enthusiast  
📧 [makadiyavatsal0205@gmail.com](mailto:makadiyavatsal0205@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/makadiyavatsall) • [GitHub](https://github.com/makadiyavatsall)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:F59E0B,100:FACC15&height=120&section=footer&text=AI%20Building%20Approval%20Prediction%20System%20🏗️&fontSize=20&fontColor=000000" />
</p>


---



