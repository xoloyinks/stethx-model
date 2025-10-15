# 🩺 DSS for Early Diabetes Detection and Prevention  

## 📘 Overview  
This project is a **Decision Support System (DSS)** that integrates **Machine Learning (ML)** and **Large Language Models (LLMs)** to assist in the **early detection and prevention of diabetes**.  
The system predicts diabetic risk based on patient health metrics and provides contextual recommendations to address comorbidities and preventive measures.  

## ⚙️ Methodology  
- **Machine Learning Model:** Logistic Regression  
- **Dataset:** Publicly available Kaggle diabetes dataset (100,000 records)  
- **Input Features:** Age, BMI, blood glucose level, HbA1c, smoking history, hypertension status, and other lifestyle factors  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **LLM Integration:** Provides interpretive analysis of model outcomes, identifies key risk factors, and suggests prevention strategies and comorbidity insights  

## 📊 Results  
| Metric | Score |
|---------|--------|
| Accuracy | 0.96 |
| Precision | 0.96 (Class 0) / 0.86 (Class 1) |
| Recall | 0.99 (Class 0) / 0.61 (Class 1) |
| F1 Score | 0.98 (Class 0) / 0.72 (Class 1) |
| ROC-AUC | 0.96 |

The Logistic Regression model achieved strong overall accuracy and reliability, demonstrating potential for clinical application in diabetes risk assessment.

## 🧠 System Workflow  
1. **User Input:** Health parameters such as age, BMI, glucose levels, and lifestyle habits  
2. **Prediction:** ML model predicts diabetic or non-diabetic status  
3. **Analysis:** LLM interprets results, identifies risk factors and comorbidities  
4. **Recommendation:** Personalized guidance for prevention and management  

## 🧩 Tech Stack  
- **Frontend:** Next.js (or your UI framework, if applicable)  
- **Backend:** Python (Flask / FastAPI)  
- **ML Libraries:** Scikit-learn, Pandas, NumPy  
- **LLM Integration:** OpenAI / Gemini API (depending on your setup)  
- **Visualization:** Matplotlib / Seaborn  


