# Vedant-Bank-Churn-Analysis
This project focuses on **analyzing customer churn in a bank** — identifying key factors that lead customers to leave the bank and building a **machine learning model** to predict churn probability.   By understanding churn behavior, banks can take proactive steps to improve customer retention and profitability.
## Objectives
- Perform **Exploratory Data Analysis (EDA)** to understand customer demographics and behavior.
- Identify patterns and features contributing to customer churn.
- Build and evaluate predictive models to forecast churn.
- Provide actionable insights and data-driven recommendations for reducing churn.

---

## Dataset Information
- **Dataset Name:** <a href=https://github.com/Vedant2331/Vedant-Bank-Churn-Analysis/blob/main/Churn_Modelling.csv>Churn_Modelling</a>
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

## Machine Learning Workflow

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

## Technologies Used
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

## Key Insights
- Older customers and those with **lower credit scores** have higher churn rates.  
- Customers with **fewer bank products** or **inactive accounts** are more likely to leave.  
- **France** showed a relatively higher churn percentage compared to other regions.

---

## Results
- **Best Model:** XGBoost Classifier  
- **Training Accuracy:** ~86%  
- **Testing Accuracy:** ~84%  
- Generated churn probability scores for each customer and exported results to Excel.

---

## Future Improvements
- Integrate **real-time prediction API** for live customer churn monitoring.
- Implement **advanced feature selection** and **hyperparameter optimization**.
- Incorporate **customer feedback sentiment analysis** for richer insights.

---

## Project Demo
Streamlit App Preview   
<img width="1890" height="857" alt="Screenshot 2025-11-04 031639" src="https://github.com/user-attachments/assets/84d0d2d6-098a-4ce0-af1d-8d0a67d10cd9" />
<img width="1876" height="893" alt="Screenshot 2025-11-04 031710" src="https://github.com/user-attachments/assets/1715cd90-5080-470f-9f17-fe97beefe9f0" />



---

## Installation & Setup

1. **Clone the repository**
-<a href=https://github.com/Vedant2331/Vedant-Bank-Churn-Analysis>Bank-Churn-Analysis</a>
2. **Install dependencies**
pip install -r requirements.txt
3. **Run the Streamlit app**
streamlit run app.py


**Requirements**
Create a requirements.txt file with the following:
- nginx
- Copy code
- pandas
- numpy
- seaborn
- matplotlib
- plotly
- scikit-learn
- imbalanced-learn
- xgboost
- streamlit
- graphviz
- openpyxl

## Power BI dashboard
- 
