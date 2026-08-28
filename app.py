from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import warnings
from deep_translator import GoogleTranslator
from fuzzywuzzy import process
from PIL import Image
import pytesseract
import io

warnings.filterwarnings("ignore")
app = Flask(__name__)

# Load Machine Learning Model and Symptoms Dataset
try:
    model = pickle.load(open('model.pkl', 'rb'))
    dataset_symptoms = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model files: {e}")
    dataset_symptoms = []

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
            return jsonify({'error': 'For an accurate AI diagnosis, please provide at least 3 symptoms (e.g., Fever, Headache, Nausea).'})
            
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                index = dataset_symptoms.index(symptom)
                input_features[index] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).replace('_', ' ').title()
        
        description = f"Based on our NLP analysis, the model indicates a possibility of {disease_name}. Please consult a doctor."
        precautions = ["Rest adequately", "Stay hydrated", "Consult a certified physician"]
        
        return jsonify({
            'disease': disease_name,
            'description': description,
            'precautions': precautions
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal Server Error. Check server logs.'})

# --- NEW: OCR MEDICINE SCANNER ROUTE ---
@app.route('/scan_medicine', methods=['POST'])
def scan_medicine():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded. Please select a photo.'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file submitted.'})
            
        # Open image using Pillow
        img = Image.open(file.stream)
        
        # Use Tesseract AI to extract text from the image
        extracted_text = pytesseract.image_to_string(img)
        clean_text = " ".join(extracted_text.split())
        
        if not clean_text:
            return jsonify({'error': 'No text detected. Please upload a clearer photo of the medicine.'})
            
        return jsonify({'text': clean_text})
        
    except Exception as e:
        return jsonify({'error': f"OCR Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)