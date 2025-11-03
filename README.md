# Vedant-Bank-Churn-Analysis
This project focuses on **analyzing customer churn in a bank** — identifying key factors that lead customers to leave the bank and building a **machine learning model** to predict churn probability.   By understanding churn behavior, banks can take proactive steps to improve customer retention and profitability.
## 🎯 Objectives
- Perform **Exploratory Data Analysis (EDA)** to understand customer demographics and behavior.
- Identify patterns and features contributing to customer churn.
- Build and evaluate predictive models to forecast churn.
- Provide actionable insights and data-driven recommendations for reducing churn.

---

## 📁 Dataset Information
- **Dataset Name:** `Churn_Modelling.csv`
- **Source:** Kaggle / Open Source
- **Rows:** 10,000  
- **Columns:** 14  
- **Target Variable:** `Exited` (1 = Customer left the bank, 0 = Customer retained)

**Key Features:**
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

---

## 🧠 Machine Learning Workflow

### 1. Data Preprocessing
- Handled missing values and outliers.
- Encoded categorical features using **Label Encoding / One-Hot Encoding**.
- Scaled numerical features using **MinMaxScaler**.

### 2. Exploratory Data Analysis (EDA)
- Used **Seaborn**, **Matplotlib**, and **Plotly** for visual insights.
- Analyzed churn distribution by gender, geography, and customer activity.
- Visualized correlations and important numerical trends.

### 3. Handling Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** from `imblearn` to balance churn vs. non-churn data.

### 4. Model Building
Trained and compared multiple models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest**
- **XGBoost Classifier** *(selected as best-performing model)*

### 5. Model Evaluation
Metrics used:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Score

### 6. Model Deployment
- Developed an interactive **Streamlit web app** for churn prediction.
- Visualized feature importance, churn probabilities, and customer insights.

---

## 🧩 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Data Balancing | Imbalanced-learn (SMOTE) |
| App Deployment | Streamlit |
| Export | OpenPyXL, Graphviz |

---

## 📈 Key Insights
- Older customers and those with **lower credit scores** have higher churn rates.  
- Customers with **fewer bank products** or **inactive accounts** are more likely to leave.  
- **France** showed a relatively higher churn percentage compared to other regions.

---

## 🚀 Results
- **Best Model:** XGBoost Classifier  
- **Training Accuracy:** ~86%  
- **Testing Accuracy:** ~84%  
- Generated churn probability scores for each customer and exported results to Excel.

---

## 💡 Future Improvements
- Integrate **real-time prediction API** for live customer churn monitoring.
- Implement **advanced feature selection** and **hyperparameter optimization**.
- Incorporate **customer feedback sentiment analysis** for richer insights.

---

## 📸 Project Demo
Streamlit App Preview 👇  
![Bank Churn Dashboard](demo_screenshot.png)

---

## 🧰 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/bank-churn-analysis.git
   cd bank-churn-analysis
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app

bash
Copy code
streamlit run app.py
Access the app

Open your browser and go to: http://localhost:8501

📜 Requirements
Create a requirements.txt file with the following:

nginx
Copy code
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
imbalanced-learn
xgboost
streamlit
graphviz
openpyxl
