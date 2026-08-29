from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
import requests
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

# Common Hindi/Colloquial Term Mapping
LOCAL_TERM_MAP = {
    "सर दर्द": "headache", "sirdard": "headache", "sir dard": "headache",
    "bukhar": "fever", "बुखार": "fever", "tap": "fever",
    "pet dard": "stomach_pain", "पेट दर्द": "stomach_pain",
    "khasi": "cough", "खांसी": "cough", "khokla": "cough",
    "thakan": "fatigue", "थकान": "fatigue", "thakwa": "fatigue",
    "ulti": "vomiting", "उल्टी": "vomiting",
    "dast": "diarrhoea", "दस्त": "diarrhoea", "loose motion": "diarrhoea", "लेटरींग": "diarrhoea", "latrine": "diarrhoea",
    "chills": "chills", "thand": "chills", "ठंड": "chills",
    "khujli": "itching", "खुजली": "itching"
}

# --- 2. Advanced NLP Translation & Extraction Layer ---
def process_nlp_symptoms(raw_symptoms_list):
    final_symptoms = []
    combined_raw_text = " ".join([str(s) for s in raw_symptoms_list]).lower()
    
    # 1. Local phrase override
    for phrase, mapped_salt in LOCAL_TERM_MAP.items():
        if phrase in combined_raw_text:
            final_symptoms.append(mapped_salt)
            
    # 2. Translation
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(combined_raw_text).lower()
    except:
        translated_text = combined_raw_text
        
    # Check dataset terms directly against translated text
    for symp in dataset_symptoms:
        clean_symp_name = symp.replace('_', ' ').lower()
        if clean_symp_name in translated_text or clean_symp_name in combined_raw_text:
            final_symptoms.append(symp)
            
    # 3. Word-by-word fuzzy fallback
    words = translated_text.split()
    for word in words:
        if len(word) >= 4 and dataset_symptoms:
            match, score = process.extractOne(word, dataset_symptoms)
            if score > 75:
                final_symptoms.append(match)
                
    return list(set(final_symptoms))

# --- 3. Medicine Knowledge Base ---
MEDICINE_DATABASE = {
    "paracetamol": {
        "name": "Paracetamol / Paracip-500 / Dolo 650",
        "search_term": "Paracetamol",
        "aliases": ["paracip", "dolo", "calpol", "crocin", "acetaminophen", "pacimol", "febrex", "wracip", "poeromo", "tablets ip 500", "500 mg"],
        "generic_info": "Generic Salt: Paracetamol IP (500mg / 650mg). Certified Essential Drug.",
        "composition": "Paracetamol IP (500mg / 650mg) - Pure Analgesic & Antipyretic Agent",
        "usage": "Relief from high/mild fever, severe tension headache, joint pain, toothache, and body pain.",
        "safety_note": "Maximum daily dose is 4000mg. Keep 4 to 6 hours gap between doses. Avoid alcohol.",
        "is_essential": True
    },
    "azithromycin": {
        "name": "Azithromycin 500mg (Azithral / Azee)",
        "search_term": "Azithromycin",
        "aliases": ["azithral", "azee", "azimax", "zady", "azibact", "azithro", "500 tab"],
        "generic_info": "Generic Salt: Azithromycin 500mg Tablet IP (NLEM Certified).",
        "composition": "Azithromycin Dihydrate (500mg) - Broad Spectrum Antibiotic",
        "usage": "Treats bacterial throat infections, chest infections, and sinus issues.",
        "safety_note": "Complete the full 3 or 5-day course. Take 1 hr before or 2 hrs after food.",
        "is_essential": True
    },
    "cetirizine": {
        "name": "Cetirizine 10mg (Okacet / Cetzine)",
        "search_term": "Cetirizine",
        "aliases": ["okacet", "cetzine", "alerid", "zyrtec", "cetriz", "cetirizine hydrochloride"],
        "generic_info": "Generic Salt: Cetirizine HCl 10mg. Standard OTC Anti-allergic.",
        "composition": "Cetirizine Hydrochloride IP (10mg) - Antihistamine",
        "usage": "Relieves allergic sneezing, runny nose, watery eyes, and skin itching.",
        "safety_note": "May cause mild drowsiness. Avoid driving immediately after consumption.",
        "is_essential": True
    },
    "pantoprazole": {
        "name": "Pantoprazole 40mg (Pan 40 / Pantocid)",
        "search_term": "Pantoprazole",
        "aliases": ["pan40", "pantocid", "pantosec", "pan-d", "pantoprazole gastro", "pantodac"],
        "generic_info": "Generic Salt: Pantoprazole Gastro-Resistant Tablets IP (40mg).",
        "composition": "Pantoprazole Sodium (40mg) - Proton Pump Inhibitor (PPI)",
        "usage": "Controls acidity, GERD, heartburn, peptic ulcers, and stomach burn.",
        "safety_note": "Consume once daily in the morning, 30 minutes before food.",
        "is_essential": True
    },
    "combiflam": {
        "name": "Combiflam / Flexon",
        "search_term": "Ibuprofen Paracetamol",
        "aliases": ["flexon", "brufen", "ibuprofen", "ibugesic", "ibuprofen paracetamol"],
        "generic_info": "Generic Salt: Ibuprofen 400mg + Paracetamol 325mg Combination.",
        "composition": "Ibuprofen (400mg) + Paracetamol (325mg) - Dual Action NSAID",
        "usage": "Potent relief for acute muscular pain, dental pain, and sprains.",
        "safety_note": "Always consume strictly after meals. Avoid if having kidney issues.",
        "is_essential": False
    }
}

# --- 4. Application Routes ---

@app.route('/')
def home():
    return render_template('index.html', symptoms=dataset_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        frontend_symptoms = data.get('symptoms', [])
        
        if not frontend_symptoms:
            return jsonify({'error': 'Please describe or select at least one symptom.'})
            
        smart_symptoms = process_nlp_symptoms(frontend_symptoms)
        
        if not smart_symptoms:
            return jsonify({'error': 'System could not identify specific medical symptoms from your input. Please try specific terms like "bukhar", "sirdard", "diarrhoea", etc.'})
            
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                idx = dataset_symptoms.index(symptom)
                input_features[idx] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).strip()
        
        disease_desc = f"Clinical pattern matched {len(smart_symptoms)} identified symptom(s): {', '.join([s.replace('_',' ').title() for s in smart_symptoms])}."
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
            precautions = ["Rest adequately and monitor body vitals", "Maintain hydration with ORS/Water", "Consult a registered doctor"]
            
        return jsonify({
            'disease': disease_name.replace('_', ' ').title(),
            'description': disease_desc,
            'precautions': precautions
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': f'Diagnosis error: {str(e)}'})

@app.route('/scan-medicine', methods=['POST'])
def scan_medicine():
    try:
        data = request.get_json() or {}
        raw_text = data.get('raw_text', '').lower()
        
        if not raw_text or len(raw_text.strip()) < 2:
            return jsonify({'error': 'No readable text received from scanner.'})
            
        matched_med = None
        best_overall_score = 0
        
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
                'usage': 'Prescription drug. Follow medical practitioner advice.',
                'safety_note': 'Ensure batch number and expiry date are verified before consuming.'
            })
            
    except Exception as e:
        return jsonify({'error': f'Scanner processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)