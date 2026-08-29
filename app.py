from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
from deep_translator import GoogleTranslator
from fuzzywuzzy import process

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. Load ML Model & Datasets ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    dataset_symptoms = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    print(f"Model loading error: {e}")
    dataset_symptoms = []

try:
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
except Exception as e:
    print(f"CSV loading error: {e}")
    description_df = pd.DataFrame()
    precaution_df = pd.DataFrame()

# --- 2. NLP Translation & Matching Layer ---
def process_nlp_symptoms(raw_symptoms_list):
    final_symptoms = []
    translator = GoogleTranslator(source='auto', target='en')
    
    for raw_word in raw_symptoms_list:
        raw_word = str(raw_word).strip()
        if len(raw_word) < 2:
            continue
        try:
            english_word = translator.translate(raw_word).lower()
        except:
            english_word = raw_word.lower()
            
        if english_word in dataset_symptoms:
            final_symptoms.append(english_word)
        else:
            if dataset_symptoms:
                best_match, score = process.extractOne(english_word, dataset_symptoms)
                if score > 70:
                    final_symptoms.append(best_match)
                    
    return list(set(final_symptoms))

# --- 3. Routes ---

@app.route('/')
def home():
    return render_template('index.html', symptoms=dataset_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        frontend_symptoms = data.get('symptoms', [])
        
        if not frontend_symptoms:
            return jsonify({'error': 'Please provide at least one symptom.'})
            
        smart_symptoms = process_nlp_symptoms(frontend_symptoms)
        
        if not smart_symptoms:
            return jsonify({'error': 'System could not match your symptoms. Please try specific terms.'})
            
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                idx = dataset_symptoms.index(symptom)
                input_features[idx] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).strip()
        
        # Fetch dynamic description from CSV
        disease_desc = "Clinical assessment pattern matching completed."
        if not description_df.empty and 'Disease' in description_df.columns:
            match = description_df[description_df['Disease'].str.lower() == disease_name.lower()]
            if not match.empty:
                disease_desc = match.iloc[0]['Description']
                
        # Fetch dynamic precautions from CSV
        precautions = []
        if not precaution_df.empty and 'Disease' in precaution_df.columns:
            match_prec = precaution_df[precaution_df['Disease'].str.lower() == disease_name.lower()]
            if not match_prec.empty:
                row = match_prec.iloc[0]
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in row and pd.notna(row[col]):
                        precautions.append(str(row[col]).title())
                        
        if not precautions:
            precautions = ["Rest adequately", "Maintain proper hydration", "Consult a certified doctor for validation"]
            
        return jsonify({
            'disease': disease_name.replace('_', ' ').title(),
            'description': disease_desc,
            'precautions': precautions
        })
    except Exception as e:
        print(f"Prediction logic error: {e}")
        return jsonify({'error': 'Internal server error while evaluating data.'})

@app.route('/scan-medicine', methods=['POST'])
def scan_medicine():
    try:
        data = request.get_json()
        raw_text = data.get('raw_text', '').lower()
        
        if not raw_text:
            return jsonify({'error': 'No text detected from image.'})
            
        if any(term in raw_text for term in ['paracetamol', 'dolo', 'calpol', 'acetaminophen', 'crocin']):
            medicine_data = {
                'medicine_name': 'Paracetamol / Dolo 650',
                'composition': 'Paracetamol IP (650mg / 500mg) - Analgesic & Antipyretic',
                'usage': 'Widely prescribed for relieving mild to moderate fever, headaches, body aches, and post-viral recovery.'
            }
        elif any(term in raw_text for term in ['azithromycin', 'azithral', 'azee']):
            medicine_data = {
                'medicine_name': 'Azithromycin (500mg)',
                'composition': 'Azithromycin Dihydrate - Macrolide Antibiotic',
                'usage': 'Used in treatment of bacterial respiratory tract infections, tonsillitis, and ear/throat infections.'
            }
        elif any(term in raw_text for term in ['cetirizine', 'cetzine', 'okacet', 'alerid']):
            medicine_data = {
                'medicine_name': 'Cetirizine Hydrochloride',
                'composition': 'Cetirizine HCl (10mg) - Second-Generation Antihistamine',
                'usage': 'Used for allergic rhinitis, cold symptoms, watery eyes, sneezing, and skin itching/urticaria.'
            }
        elif any(term in raw_text for term in ['pantoprazole', 'pan', 'pantocid', 'pantosec']):
            medicine_data = {
                'medicine_name': 'Pantoprazole (40mg)',
                'composition': 'Pantoprazole Sodium - Proton Pump Inhibitor (PPI)',
                'usage': 'Reduces excess stomach acid production; treats gastroesophageal reflux disease (GERD) and gastritis.'
            }
        elif any(term in raw_text for term in ['ibuprofen', 'brufen', 'combiflam']):
            medicine_data = {
                'medicine_name': 'Ibuprofen / Combiflam',
                'composition': 'Ibuprofen (400mg) + Paracetamol (325mg) - NSAID',
                'usage': 'Relieves acute inflammatory pain, dental pain, muscular discomfort, and joint pain.'
            }
        else:
            medicine_data = {
                'medicine_name': 'Detected Clinical Formulation',
                'composition': f"Extracted Text Fragment: {raw_text[:120].strip()}",
                'usage': 'Salt/formulation identified. Please consult a registered medical practitioner before consumption.'
            }
            
        return jsonify(medicine_data)
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)