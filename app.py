from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
import json
import os
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

# --- 2. Dynamic JSON Symptoms Dictionary Loader ---
LOCAL_TERM_MAP = {}
if os.path.exists('symptoms_map.json'):
    try:
        with open('symptoms_map.json', 'r', encoding='utf-8') as f:
            LOCAL_TERM_MAP = json.load(f)
    except Exception as e:
        print(f"JSON Load error: {e}")

# Fallback basic terms if json missing
if not LOCAL_TERM_MAP:
    LOCAL_TERM_MAP = {
        "bukhar": "high_fever", "बुखार": "high_fever", "fever": "high_fever",
        "sirdard": "headache", "सर दर्द": "headache", "headache": "headache",
        "sardi": "cold", "सर्दी": "cold", "khasi": "cough", "खांसी": "cough",
        "ulti": "vomiting", "dast": "diarrhoea", "thakan": "fatigue"
    }

# --- 3. Medicine Knowledge Base ---
MEDICINE_DATABASE = {
    "paracetamol": {
        "name": "Paracetamol (Dolo 650 / Paracip 500 / Calpol)",
        "aliases": ["paracip", "dolo", "calpol", "crocin", "acetaminophen", "pacimol", "febrex", "wracip", "poeromo", "tablets ip 500", "500 mg", "650 mg"],
        "generic_info": "Generic Salt: Paracetamol IP (500mg / 650mg). Available at PM Jan Aushadhi Kendras.",
        "composition": "Paracetamol IP - Pure Analgesic & Antipyretic Agent",
        "usage": "Relief from mild to high fever, tension headache, body ache, and toothache.",
        "safety_note": "Maximum daily limit is 4000mg. Keep a 4-6 hour gap between doses. Avoid alcohol."
    },
    "amoxicillin": {
        "name": "Amoxicillin + Clavulanic Acid (Augmentin 625 / Moxikind-CV)",
        "aliases": ["augmentin", "moxikind", "moxclav", "clavmox", "amoxyclav", "625 duo", "amoxicillin"],
        "generic_info": "Generic Salt: Amoxicillin (500mg) + Potassium Clavulanate (125mg) Tablet IP.",
        "composition": "Penicillin Class Broad Spectrum Antibacterial + Beta-lactamase Inhibitor",
        "usage": "Treats severe respiratory infections, ear-nose-throat infections, dental abscess, and UTI.",
        "safety_note": "Prescription antibiotic. Complete the full prescribed course strictly after meals."
    },
    "azithromycin": {
        "name": "Azithromycin (Azithral 500 / Azee 500)",
        "aliases": ["azithral", "azee", "azimax", "zady", "azibact", "azithro", "azithromycin 500"],
        "generic_info": "Generic Salt: Azithromycin 500mg Tablet IP (NLEM Listed Generic).",
        "composition": "Macrolide Broad Spectrum Antibacterial Agent",
        "usage": "Treats bacterial throat tonsillitis, chest infections, sinusitis, and skin infections.",
        "safety_note": "Take 1 hour before or 2 hours after food. Consume once daily for 3 to 5 days."
    },
    "montelukast": {
        "name": "Montelukast + Levocetirizine (Montair-LC / Montek-LC)",
        "aliases": ["montair", "montek", "levocet", "monticope", "telekast", "montair-lc", "montek-lc"],
        "generic_info": "Generic Salt: Montelukast Sodium (10mg) + Levocetirizine HCl (5mg).",
        "composition": "Leukotriene Receptor Antagonist + Non-sedating Antihistamine",
        "usage": "Relief from allergic asthma, chronic sneezing, allergic rhinitis, and night-time coughing.",
        "safety_note": "Best taken at bedtime as it may cause mild relaxation/sleepiness."
    },
    "cetirizine": {
        "name": "Cetirizine 10mg (Okacet / Cetzine / Alerid)",
        "aliases": ["okacet", "cetzine", "alerid", "zyrtec", "cetriz", "cetirizine"],
        "generic_info": "Generic Salt: Cetirizine Hydrochloride IP 10mg.",
        "composition": "Second-Generation Antihistaminic Agent",
        "usage": "Relieves runny nose, watery eyes, urticaria, skin itching, and dust allergy.",
        "safety_note": "May cause mild drowsiness. Avoid driving immediately after consumption."
    },
    "pantoprazole": {
        "name": "Pantoprazole 40mg (Pan 40 / Pantocid / Pantodac)",
        "aliases": ["pan 40", "pan40", "pantocid", "pantosec", "pantodac", "pantoprazole"],
        "generic_info": "Generic Salt: Pantoprazole Gastro-Resistant Tablets IP 40mg.",
        "composition": "Proton Pump Inhibitor (Gastric Acid Reducer)",
        "usage": "Controls severe gastric acidity, GERD, heartburn, stomach ulcers, and acid reflux.",
        "safety_note": "Take once daily in the morning, 30 minutes before your first meal/breakfast."
    },
    "combiflam": {
        "name": "Ibuprofen + Paracetamol (Combiflam / Flexon)",
        "aliases": ["combiflam", "flexon", "brufen", "ibugesic plus", "ibuprofen paracetamol"],
        "generic_info": "Generic Salt: Ibuprofen (400mg) + Paracetamol (325mg) Tablet IP.",
        "composition": "Dual-action NSAID Analgesic & Anti-inflammatory",
        "usage": "Acute muscular pain, sprains, dental surgery pain, and joint swelling.",
        "safety_note": "Always consume strictly after a full meal to prevent stomach gastric irritation."
    }
}

# --- 4. NLP Symptom Extractor ---
def process_nlp_symptoms(raw_symptoms_list):
    extracted_tags = []
    combined_text = " ".join([str(s) for s in raw_symptoms_list]).lower()
    
    # 1. Match from Local/JSON Map
    for phrase, tag in LOCAL_TERM_MAP.items():
        if phrase in combined_text:
            extracted_tags.append(tag)
            
    # 2. Translation layer
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(combined_text).lower()
    except:
        translated = combined_text
        
    for phrase, tag in LOCAL_TERM_MAP.items():
        if phrase in translated:
            extracted_tags.append(tag)
            
    # 3. Direct match with dataset features
    for symp in dataset_symptoms:
        clean_name = symp.replace('_', ' ').lower()
        if clean_name in translated or clean_name in combined_text:
            extracted_tags.append(symp)
            
    return list(set(extracted_tags)), combined_text

# --- 5. App Routes ---

@app.route('/')
def home():
    return render_template('index.html', symptoms=dataset_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        frontend_symptoms = data.get('symptoms', [])
        
        if not frontend_symptoms:
            return jsonify({'error': 'Kripya kam se kam ek symptom likhein ya select karein.'})
            
        smart_symptoms, raw_user_text = process_nlp_symptoms(frontend_symptoms)
        
        # --- A. CLINICAL CLUSTER TRIAGE LAYER ---
        has_fever = any(s in smart_symptoms for s in ['high_fever', 'mild_fever']) or any(w in raw_user_text for w in ['bukhar', 'बुखार', 'fever', 'tap'])
        has_cold = any(s in smart_symptoms for s in ['cold', 'cough', 'continuous_sneezing', 'throat_irritation', 'runny_nose']) or any(w in raw_user_text for w in ['sardi', 'सर्दी', 'khasi', 'cold', 'jukham'])
        has_headache = 'headache' in smart_symptoms or any(w in raw_user_text for w in ['sirdard', 'सर दर्द', 'headache'])
        has_loose_motion = any(s in smart_symptoms for s in ['diarrhoea', 'vomiting', 'stomach_pain']) or any(w in raw_user_text for w in ['dast', 'loose motion', 'ulti', 'pet kharab'])

        # Case 1: Fever + Cold/Cough
        if has_fever and has_cold:
            return jsonify({
                'disease': 'Seasonal Influenza / Common Viral Flu',
                'description': 'Bukhar ke sath sardi, khasi aur gale me kharash aam viral respiratory infection (Flu) ke lakshan hain. Yeh mausam badalne par 3 se 5 din tak rehta hai.',
                'precautions': [
                    'Paracetamol (500mg) for fever and body temperature control',
                    'Steam inhalation (Bhaap lein) aur gungune paani se namak daal kar gargle karein',
                    'Garm paani, soup aur liquid fluids zyada matra me lein',
                    'Agar bukhar 102°F se upar jaye ya 4 din se zyada rahe toh doctor se milen'
                ]
            })

        # Case 2: Fever + Headache
        if has_fever and has_headache:
            return jsonify({
                'disease': 'Acute Viral Pyrexia with Cephalea',
                'description': 'Bukhar ke sath sir dard aam taur par viral infection, dehydration ya sharir me thakan ki wajah se hota hai.',
                'precautions': [
                    'Paracetamol for fever relief and rest in a calm room',
                    'ORS / Nariyal paani aur dehydration se bachein',
                    'Mobile aur laptop screen time se parhez karein',
                    'Agar ulti ya chakkar aayein toh doctor se blood test karwayein'
                ]
            })

        # Case 3: Loose motion / Vomiting
        if has_loose_motion:
            return jsonify({
                'disease': 'Acute Gastroenteritis (Stomach Infection)',
                'description': 'Dast, ulti ya pet dard aam taur par contaminated food ya paani se hone wale bacterial/viral infection ke sanket hain.',
                'precautions': [
                    'ORS (Oral Rehydration Solution) har dast ke baad piyein',
                    'Khane me dahi, khichdi aur kela jaise halki cheezein lein',
                    'Tali-bhuni aur bahar ki cheezon se bilkul parhez karein',
                    'Dehydration na hone dein aur doctor se consultation lein'
                ]
            })

        # Case 4: Single Symptom Isolated
        if len(smart_symptoms) <= 1:
            if has_fever:
                return jsonify({
                    'disease': 'Viral Pyrexia (Acute Viral Fever)',
                    'is_single': True,
                    'identified_symptom': 'Bukhar (Fever)',
                    'follow_up_question': 'Aapko bukhar ke sath inme se aur kya takleef hai?',
                    'follow_up_options': ['Sardi / Khasi (Cold/Cough)', 'Sir dard (Headache)', 'Thand lagna (Chills)', 'Ulti / Dast (Vomiting)'],
                    'description': 'Sirf bukhar aana aam viral infection ya seasonal change ka sanket hai.',
                    'precautions': [
                        'Paracetamol (500mg) as advised for fever',
                        'Pani aur ORS ka sevan badhayein',
                        'Gili patti (Cold compress) lagayein agar bukhar tez ho',
                        'Doctor se consult karein'
                    ]
                })
            elif has_cold:
                return jsonify({
                    'disease': 'Upper Respiratory Irritation / Common Cold',
                    'description': 'Sardi, khasi ya chhinke aana aam allergy ya viral common cold ka sanket hai.',
                    'precautions': [
                        'Steam inhalation (Bhaap lein)',
                        'Gunguna paani piyein aur thandi cheezon se parhez karein',
                        'Antihistamine (jaise Cetirizine 10mg) raat ko le sakte hain'
                    ]
                })
            elif has_headache:
                return jsonify({
                    'disease': 'Tension Headache / Migraine Strain',
                    'description': 'Akela sir dard neend ki kami, screen stress ya dehydration se ho sakta hai.',
                    'precautions': [
                        'Shant andhere kamre me rest karein',
                        'Khoob paani piyein',
                        'Screen time kam karein'
                    ]
                })

        # --- B. MULTI-SYMPTOM ML MODEL WITH CONFIDENCE GUARD ---
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                idx = dataset_symptoms.index(symptom)
                input_features[idx] = 1
                
        features_array = np.array([input_features])
        
        # Check model probabilities
        try:
            probabilities = model.predict_proba(features_array)[0]
            max_prob = np.max(probabilities)
            prediction = model.predict(features_array)[0]
            disease_name = str(prediction).strip()
            
            # If model confidence is very low (< 35%), do not show extreme diseases
            if max_prob < 0.35:
                disease_name = 'Non-Specific Viral Infection'
        except:
            prediction = model.predict(features_array)[0]
            disease_name = str(prediction).strip()
        
        # Blacklist critical false alarms if input is general
        if disease_name.lower() in ['aids', 'dimorphic hemmorhoids(piles)', 'hepatitis a', 'tuberculosis', 'paralysis (brain hemorrhage)']:
            if not any(k in smart_symptoms for k in ['loss_of_balance', 'unsteadiness', 'altered_sensorium', 'blood_in_sputum', 'yellowish_skin']):
                disease_name = 'Seasonal Viral Syndrome'

        disease_desc = "Clinical symptom pattern evaluation completed."
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
        return jsonify({'error': 'Diagnosis evaluation failed. Please try again.'})

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
                'generic_info': 'Generic Salt Equivalent: Verify active salt on packaging with a pharmacist.',
                'composition': f"Extracted Text: {raw_text[:120].strip()}",
                'usage': 'Prescription medication detected. Follow doctor/pharmacist dosage directions.',
                'safety_note': 'Ensure batch number and expiry date are verified before consuming.'
            })
            
    except Exception as e:
        return jsonify({'error': f'Scanner processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)