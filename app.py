from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import warnings
from deep_translator import GoogleTranslator
from fuzzywuzzy import process
import requests
import re

warnings.filterwarnings("ignore")
app = Flask(__name__)

# Load Machine Learning Model and Symptoms Dataset
try:
    model = pickle.load(open('model.pkl', 'rb'))
    dataset_symptoms = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {e}")
    dataset_symptoms = []

# --- DUMMY DATABASE FOR MEDICINE ALTERNATIVES ---
MEDICINE_DB = {
    "paracetamol": [
        {"name": "Jan Aushadhi Paracetamol 500mg", "price": "₹5.00", "manufacturer": "Generic (Govt)"},
        {"name": "Crocin 500", "price": "₹15.00", "manufacturer": "GSK"},
        {"name": "Dolo 500", "price": "₹16.00", "manufacturer": "Micro Labs"}
    ],
    "amoxicillin": [
        {"name": "Jan Aushadhi Amoxicillin 500mg", "price": "₹35.00", "manufacturer": "Generic (Govt)"},
        {"name": "Novamox 500", "price": "₹65.00", "manufacturer": "Cipla"},
        {"name": "Mox 500", "price": "₹70.00", "manufacturer": "Sun Pharma"}
    ],
    "pantoprazole": [
        {"name": "Jan Aushadhi Pantoprazole 40mg", "price": "₹15.00", "manufacturer": "Generic (Govt)"},
        {"name": "Pan 40", "price": "₹55.00", "manufacturer": "Alkem"},
        {"name": "Pantosec 40", "price": "₹60.00", "manufacturer": "Cipla"}
    ]
}

def extract_medicine_composition(ocr_text):
    text_lower = ocr_text.lower()
    for comp, alternatives in MEDICINE_DB.items():
        if comp in text_lower:
            return comp.title(), alternatives
    return None, []

# --- ADVANCED NLP LAYER ---
def process_nlp_symptoms(raw_symptoms_list):
    final_symptoms = []
    translator = GoogleTranslator(source='auto', target='en')
    for raw_word in raw_symptoms_list:
        try:
            english_word = translator.translate(raw_word).lower()
        except:
            english_word = raw_word.lower()
        
        if english_word in dataset_symptoms:
            final_symptoms.append(english_word)
        else:
            best_match, score = process.extractOne(english_word, dataset_symptoms)
            if score > 70:
                final_symptoms.append(best_match)
    return list(set(final_symptoms))

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
            return jsonify({'error': 'System could not match your words to medical terms. Please try again.'})
        
        if len(smart_symptoms) < 3:
            return jsonify({'error': 'For an accurate AI diagnosis, please provide at least 3 symptoms.'})
            
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                index = dataset_symptoms.index(symptom)
                input_features[index] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).replace('_', ' ').title()
        
        return jsonify({
            'disease': disease_name,
            'description': f"Based on our NLP analysis, the model indicates a possibility of {disease_name}.",
            'precautions': ["Rest adequately", "Stay hydrated", "Consult a certified physician"]
        })
        
    except Exception as e:
        return jsonify({'error': 'Internal Server Error. Check server logs.'})

# --- SMART OCR ALTERNATIVE SCANNER ---
@app.route('/scan_medicine', methods=['POST'])
def scan_medicine():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded. Please select a photo.'})
            
        file = request.files['file']
        
        payload = {'isOverlayRequired': False, 'apikey': 'helloworld', 'language': 'eng'}
        response = requests.post('https://api.ocr.space/parse/image',
                                 files={'file': (file.filename, file.stream, file.mimetype)},
                                 data=payload)
        
        result = response.json()
        
        if result.get('IsErroredOnProcessing') or not result.get('ParsedResults'):
            return jsonify({'error': 'No text detected. Please upload a clearer photo.'})
            
        extracted_text = result['ParsedResults'][0].get('ParsedText', '').strip()
        
        # Pass raw text to our smart function
        composition, alternatives = extract_medicine_composition(extracted_text)
        
        if not composition:
            return jsonify({'error': 'Could not identify a known medicine composition from the image. Try a clearer photo.'})
            
        return jsonify({
            'composition': composition,
            'alternatives': alternatives
        })
        
    except Exception as e:
        return jsonify({'error': f"Scanner Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)