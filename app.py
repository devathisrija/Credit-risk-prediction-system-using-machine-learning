from flask import Flask, render_template, request
import pickle, numpy as np, os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model_paths = {
    'Decision Tree': os.path.join(MODELS_DIR, 'Credit_card_model_dt.pkl'),
    'Random Forest': os.path.join(MODELS_DIR, 'Credit_card_model_rf.pkl'),
    'Naive Bayes': os.path.join(MODELS_DIR, 'Credit_card_model_nb.pkl'),
    'Logistic Regression': os.path.join(MODELS_DIR, 'Credit_card_model_lr.pkl'),
    'K Nearest Neighbor': os.path.join(MODELS_DIR, 'Credit_card_model_knn.pkl')
}
models = {name: pickle.load(open(path, 'rb')) for name, path in model_paths.items()}

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    if request.method=='POST':
        sel = request.form.get('selected_model')
        try:
            X = [
                float(request.form['RevolvingUtilizationOfUnsecuredLine']),
                int(request.form['age']),
                int(request.form['NumberOfOpenCreditLinesAndLoans']),
                int(request.form['NumberRealEstateLoansOrLines']),
                float(request.form['MonthlyIncome']),
                int(request.form['NumberOfDependents']),
                1 if request.form['Gender']=='male' else 0,
                1 if request.form['Rented_OwnHouse']=='own house' else 0
            ]
            # Region: keep 'central','east'; drop others
            X += [
                1 if request.form['Region']=='central' else 0,
                1 if request.form['Region']=='east' else 0
            ]
            # Occupation: non-officer, officer1, officer2
            X += [
                1 if request.form['Occupation']=='non-officer' else 0,
                1 if request.form['Occupation']=='officer1' else 0,
                1 if request.form['Occupation']=='officer2' else 0
            ]
            # Education: graduate, matric, phd
            X += [
                1 if request.form['Education']=='graduate' else 0,
                1 if request.form['Education']=='matric' else 0,
                1 if request.form['Education']=='phd' else 0
            ]

            if len(X)!=16:
                raise ValueError(f"Expect 16 features but got {len(X)}")

            model = models.get(sel)
            if model:
                y=model.predict(np.array([X]))[0]
                prediction = 'Good Customer' if y==1 else 'Bad Customer'
            else:
                prediction = 'Select a valid model'
        except Exception as e:
            print("Error:", e)
            prediction = 'Invalid input / feature encoding'
    return render_template('index.html', prediction=prediction, selected_model=request.form.get('selected_model'))

if __name__=='__main__':
    app.run(debug=True)

