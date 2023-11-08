
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model= joblib.load('Model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = [float(request.form['BMI']), float(request.form['Smoking']),float(request.form['AlcoholDrinking']),float(request.form['Stroke']),
                  float(request.form['PhysicalHealth']),float(request.form['MentalHealth']),float(request.form['Sex']),
                  float(request.form['AgeCategory']),float(request.form['Race']),float(request.form['Diabetic']),float(request.form['PhysicalActivity'])]
                  

   
    scaled_data = np.array([input_data])  # Assuming min-max scaling

   
    prediction = model.predict(scaled_data)
    
    if prediction == 0: 
        prediction = "good"
    if prediction == 1:
        prediction = "standard"
    if prediction== 2:
         prediction= "poor"
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=7000)



