                          ** HUPA Biomarker Analysis & Glucose Trend Dashboard**  
End‑to‑end data analytics pipeline including Data Cleaning, Descriptive Analytics, Prescriptive Insights, Predictive Modeling, and a
Streamlit Dashboard.
** Project Overview  **
This project analyzes **HUPA biomarker data** to understand glucose behavior and generate actionable insights.  
The workflow follows a structured analytics lifecycle:
1. **Category 1 – Data Cleaning**
2. **Category 2 – Descriptive Analytics**
3. **Category 3 – Prescriptive Analytics**
4. **Category 4 – Predictive Modeling**
5. **Streamlit Dashboard – Final Presentation Layer**Each stage is implemented in separate Jupyter notebooks for clarity and modularity.

 **Category 1: Data Cleaning**
** Tasks Performed**
 Removed duplicates and null values  
 Standardized column names  
 Converted timestamps to consistent datetime format  
 Cleaned biomarker ranges  
 Merged demographic + biomarker datasets  
 Handled outliers using IQR and domain rules  
** Output ** 
A clean, analysis‑ready dataset used in all downstream categories.

 **Category 2: Descriptive Analytics**
 Key Analyses
 Hourly glucose trend  
 Daily glucose trend  
 Weekly glucose trend  
 Distribution of biomarkers  
 Correlation heatmaps  
 Summary statistics  
** Insights**
 Identified peak glucose hours  
 Observed weekly glucose cycles  
 Detected biomarker patterns linked to glucose variability  

 **Category 3: Prescriptive Analytics**
**Techniques Used**
 Threshold‑based recommendations  
 Lifestyle‑based suggestions  
 Biomarker‑driven prescriptive rules  
 Risk scoring logic  
** Examples**
 High glucose + low steps → recommend physical activity  
 High HRV + stable glucose → maintain current routine  
 
 **Category 4: Predictive Modeling**
** Models Tested**
 Random Forest  
 XGBoost  
 Logistic Regression  
 Gradient Boosting  
** Target**
Predict **glucose spike risk** based on biomarkers + lifestyle features.
 Outputs
 Spike probability  
 Feature importance  
 Model evaluation metrics (AUC, F1, Recall)
 **Streamlit Dashboard**
The final output is an interactive dashboard built using **Streamlit**.
**Features**
 Upload CSV file  
 View Hourly / Daily / Weekly glucose trends  
 Auto‑parsed timestamps  
 Clean visualizations using Seaborn + Matplotlib  
 Interactive sidebar  
 Real‑time charts  



