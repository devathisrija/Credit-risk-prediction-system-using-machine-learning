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
```
credit-card/
├── app.py                     # Flask application for deployment
├── index.py                   # Model training / pipeline script
├── main code file             # Jupyter/Script for EDA & training
├── cat_to_num.py              # Script for categorical → numerical encoding
├── variable_transformation.py # Script for feature transformations
├── missing_values.py          # Script to handle missing values
├── logging_file.py            # Logging configuration
├── Credit_card_model_dt       # Saved Decision Tree model (Pickle file)
├── Credit_card_model_lr       # Saved Logistic Regression model (Pickle file)
├── Credit_card_model_nb       # Saved Naive Bayes model (Pickle file)
├── Credit_card_model_rf       # Saved Random Forest model (Pickle file)
├── Credit_card_model_knn      # Saved KNN model (Pickle file)
├── Credit_card_model1         # Another saved model file
│
├── templates/                 # HTML templates for Flask frontend
│   └── index.html             # Main HTML form for user inputs
│
├── models/                    # Folder to store trained ML models
├── __pycache__/               # Python cache files
├── .idea/                     # IDE settings (PyCharm/VSCode)
├── .venv/                     # Virtual environment
└── README.md                  # Project documentation
```
## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-risk-prediction.git
   cd credit-risk-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # On Windows
   source .venv/bin/activate       # On Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. Train and save models (if not already trained):
   ```bash
   python index.py
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

4. Enter customer details such as:
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

   and click **Predict** to get real-time creditworthiness (Good/Bad).

---

## 📊 Models Used

The following machine learning algorithms were trained and evaluated:
- K-Nearest Neighbors (KNN)  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Naive Bayes  

The best-performing model was selected based on **Accuracy, Precision, Recall, and F1-score**.

---

## 📈 Results

- Implemented a complete preprocessing pipeline (missing value handling, encoding, scaling, feature selection).  
- Achieved reliable prediction accuracy across multiple models.  
- Random Forest provided the best performance in most test cases.  
- Deployed final model for **real-time predictions** through Flask.  

---

## 📷 Screenshots

_Add screenshots of your UI and outputs here_  
Example:  

- **Homepage**  
![Homepage](screenshots/homepage.png)  

- **Prediction Result**  
![Result](screenshots/result.png)  

---

## 🛠️ Technologies Used

- **Python 3.10+**  
- **Flask** (Backend)  
- **HTML, CSS** (Frontend)  
- **Scikit-learn** (ML Models)  
- **Pandas & NumPy** (Data Processing)  
- **Matplotlib & Seaborn** (Visualization)  
- **Pickle** (Model Serialization)  

---






