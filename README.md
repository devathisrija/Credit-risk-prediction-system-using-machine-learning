# 📊 Credit Risk Prediction System using Machine Learning  

An **end-to-end Machine Learning application** that predicts customer **creditworthiness (Good/Bad)** based on financial and demographic factors.  
This project combines data preprocessing, model training, evaluation, and deployment into a seamless, user-friendly web application.  

---

## 🚀 Project Overview  
The **Credit Risk Prediction System** helps financial institutions assess whether a customer is likely to default on a loan.  
It leverages multiple machine learning algorithms and a Flask-based deployment to deliver **real-time credit risk predictions**.  

---

## 🛠️ Features  
- **Data Preprocessing Pipeline**  
  - Feature engineering and selection  
  - Handling missing values  
  - Encoding categorical variables  
  - Feature scaling  

- **Model Training & Evaluation**  
  - Algorithms: **KNN, Logistic Regression, Decision Tree, Random Forest, Naive Bayes**  
  - Trained & evaluated using **Scikit-learn pipelines**  
  - Selected best-performing model using metrics like **Accuracy, Precision, Recall, F1-score**  

- **Web Application (Flask + HTML)**  
  - User inputs:  
    - Revolving Utilization  
    - Age  
    - Open Credit Lines and Loans  
    - Real Estate Loans or Lines  
    - Monthly Income  
    - Number of Dependents  
    - Gender  
    - Region  
    - Occupation  
    - Education  
  - Outputs prediction: **Good / Bad Credit Risk**  
  - Deployed with **Pickle** for model serialization  

---

## 📂 Project Structure  

👉 Use **backticks (\`\`\`)** around the tree so Markdown renders it properly.  

---

### ✅ Option 2: Simple bullet-point style  
```markdown
## 📂 Project Structure
- data/ → Dataset (raw & processed)  
- notebooks/ → Jupyter notebooks for EDA & experiments  
- src/ → Python scripts for preprocessing & training  
- model/ → Saved trained model(s)  
- app/ → Flask application (backend + frontend)  
  - app.py → Flask backend  
  - templates/ → HTML templates  
  - static/ → CSS / JS / Images  
- requirements.txt → Dependencies  
- README.md → Project documentation  

