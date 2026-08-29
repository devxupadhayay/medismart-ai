from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
from deep_translator import GoogleTranslator
from fuzzywuzzy import process, fuzz

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

# --- 2. Advanced Medicine & Generic Database ---
MEDICINE_DATABASE = {
    "paracetamol": {
        "name": "Paracetamol / Paracip-500 / Dolo 650",
        "aliases": ["paracip", "dolo", "calpol", "crocin", "acetaminophen", "pacimol", "febrex", "wracip", "poeromo", "tablets ip 500", "500 mg"],
        "generic_info": "Generic Salt: Paracetamol IP (500mg / 650mg). Available at PM Jan Aushadhi Kendras under standard salt formulation.",
        "composition": "Paracetamol IP (500mg / 650mg) - Pure Analgesic & Antipyretic Agent",
        "usage": "Relief from high/mild fever, severe headache, joint pain, toothache, and viral body aches.",
        "safety_note": "Maximum daily dose is 4000mg. Maintain a minimum gap of 4 to 6 hours between doses. Avoid alcohol."
    },
    "azithromycin": {
        "name": "Azithromycin 500mg (Azithral / Azee)",
        "aliases": ["azithral", "azee", "azimax", "zady", "azibact", "azithro", "500 tab"],
        "generic_info": "Generic Salt: Azithromycin 500mg Tablet IP. Supplied under National Essential Medicine List (NLEM).",
        "composition": "Azithromycin Dihydrate (500mg) - Broad Spectrum Macrolide Antibiotic",
        "usage": "Treats bacterial throat infections (tonsillitis), chest infections, pneumonia, and severe sinus issues.",
        "safety_note": "Complete the full 3 or 5-day course as prescribed. Take 1 hour before or 2 hours after food."
    },
    "cetirizine": {
        "name": "Cetirizine 10mg (Okacet / Cetzine)",
        "aliases": ["okacet", "cetzine", "alerid", "zyrtec", "cetriz", "cetirizine hydrochloride"],
        "generic_info": "Generic Salt: Cetirizine HCl 10mg. Unbranded generic strips offer the identical antihistamine relief.",
        "composition": "Cetirizine Hydrochloride IP (10mg) - Second-Generation Antihistamine",
        "usage": "Relieves allergic sneezing, runny nose, allergic rhinitis, watery eyes, and skin urticaria/itching.",
        "safety_note": "May cause mild drowsiness. Avoid driving or operating heavy machinery after consumption."
    },
    "pantoprazole": {
        "name": "Pantoprazole 40mg (Pan 40 / Pantocid)",
        "aliases": ["pan40", "pantocid", "pantosec", "pan-d", "pantoprazole gastro", "pantodac"],
        "generic_info": "Generic Salt: Pantoprazole Gastro-Resistant Tablets IP (40mg). Available across generic pharmacies.",
        "composition": "Pantoprazole Sodium (40mg) - Proton Pump Inhibitor (Acid Reducer)",
        "usage": "Controls severe acidity, GERD, heartburn, peptic ulcers, and protects stomach lining.",
        "safety_note": "Recommended to consume once daily in the morning, 30 minutes before breakfast."
    },
    "combiflam": {
        "name": "Combiflam / Flexon",
        "aliases": ["flexon", "brufen", "ibuprofen", "ibugesic", "ibuprofen paracetamol"],
        "generic_info": "Generic Salt: Ibuprofen 400mg + Paracetamol 325mg Combination. Standard generic NSAID formulation.",
        "composition": "Ibuprofen (400mg) + Paracetamol (325mg) - Dual Action Anti-inflammatory & Pain Reliever",
        "usage": "Relief for acute muscular pain, joint inflammation, dental pain, and sprains.",
        "safety_note": "Always consume strictly after a full meal to prevent stomach irritation. Avoid if having renal issues."
    }
}

# --- 3. NLP Symptoms Translation Layer ---
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

# --- 4. Application Routes ---

@app.route('/')
def home():
    return render_template('index.html', symptoms=dataset_symptoms)

@app.route('/scanner')
def scanner():
    return render_template('scanner.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        frontend_symptoms = data.get('symptoms', [])
        
        if not frontend_symptoms:
            return jsonify({'error': 'Please provide at least one symptom.'})
            
        smart_symptoms = process_nlp_symptoms(frontend_symptoms)
        
        if not smart_symptoms:
            return jsonify({'error': 'Could not match symptoms with dataset. Please try specific terms.'})
            
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                idx = dataset_symptoms.index(symptom)
                input_features[idx] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).strip()
        
        disease_desc = "Clinical diagnosis pattern matching completed."
        if not description_df.empty and 'Disease' in description_df.columns:
            match = description_df[description_df['Disease'].str.lower() == disease_name.lower()]
            if not match.empty:
                disease_desc = match.iloc[0]['Description']
                
        precautions = []
        if not precaution_df.empty and 'Disease' in precaution_df.columns:
            match_prec = precaution_df[precaution_df['Disease'].str.lower() == disease_name.lower()]
            if not match_prec.empty:
                row = match_prec.iloc[0]
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in row and pd.notna(row[col]):
                        precautions.append(str(row[col]).title())
                        
        if not precautions:
            precautions = ["Rest adequately", "Ensure proper hydration", "Consult a registered doctor"]
            
        return jsonify({
            'disease': disease_name.replace('_', ' ').title(),
            'description': disease_desc,
            'precautions': precautions
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'Internal server error while evaluating health scan.'})

@app.route('/scan-medicine', methods=['POST'])
def scan_medicine():
    try:
        data = request.get_json()
        raw_text = data.get('raw_text', '').lower()
        
        if not raw_text or len(raw_text.strip()) < 2:
            return jsonify({'error': 'No readable text received from scanner.'})
            
        matched_med = None
        best_overall_score = 0
        
        # Exact keyword & alias match
        for key, details in MEDICINE_DATABASE.items():
            if key in raw_text:
                matched_med = details
                break
            for alias in details['aliases']:
                if alias in raw_text:
                    matched_med = details
                    break
            if matched_med:
                break
                
        # Fuzzy match fallback for OCR broken spellings
        if not matched_med:
            words = [w for w in raw_text.split() if len(w) >= 3]
            for word in words:
                for key, details in MEDICINE_DATABASE.items():
                    for alias in [key] + details['aliases']:
                        score = fuzz.partial_ratio(word, alias)
                        if score > best_overall_score and score >= 60:
                            best_overall_score = score
                            matched_med = details
                            
        if matched_med:
            return jsonify({
                'medicine_name': matched_med['name'],
                'generic_info': matched_med['generic_info'],
                'composition': matched_med['composition'],
                'usage': matched_med['usage'],
                'safety_note': matched_med['safety_note']
            })
        else:
            return jsonify({
                'medicine_name': 'Clinical Medicine Formulation',
                'generic_info': 'Generic Salt Equivalent: Consult your pharmacist for the non-branded chemical salt equivalent.',
                'composition': f"Extracted Signature: {raw_text[:120].strip()}",
                'usage': 'Prescription medication detected. Please verify complete packaging details with a pharmacist.',
                'safety_note': 'Ensure batch number and expiry date are verified before use.'
            })
            
    except Exception as e:
        print(f"Scanner error: {e}")
        return jsonify({'error': f'Scanner processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)