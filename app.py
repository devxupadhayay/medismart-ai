from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Data aur Model Load kar rahe hain
model = pickle.load(open('model.pkl', 'rb'))
symptoms_list = pickle.load(open('columns.pkl', 'rb'))
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')

# Jab koi website open karega, toh index.html dikhega
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)

# Yeh route JS se data lega aur prediction wapas bhejega
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    selected_symptoms = data.get('symptoms', [])
    
    if not selected_symptoms:
         return jsonify({'error': 'Please select symptoms'})

    # Model input ready karna
    input_data = np.zeros(len(symptoms_list))
    for symp in selected_symptoms:
        if symp in symptoms_list:
            index = symptoms_list.index(symp)
            input_data[index] = 1
            
    # Prediction
    prediction = model.predict([input_data])[0]
    
    # Description
    desc_col = description_df.columns[1]
    desc_val = description_df[description_df['Disease'] == prediction][desc_col].values
    desc = desc_val[0] if len(desc_val) > 0 else "Description not available."
    
    # Precautions
    prec_val = precaution_df[precaution_df['Disease'] == prediction].values
    precautions = []
    if len(prec_val) > 0:
        for i in range(1, 5):
            if pd.notna(prec_val[0][i]):
                precautions.append(str(prec_val[0][i]).title())

    # JSON format mein HTML/JS ko result bhej rahe hain
    return jsonify({
        'disease': str(prediction).upper(),
        'description': str(desc),
        'precautions': precautions
    })

if __name__ == '__main__':
    app.run(debug=True)