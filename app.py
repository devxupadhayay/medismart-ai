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

# Common Indian colloquial mappings
LOCAL_TERM_MAP = {
    "सर दर्द": "headache", "sirdard": "headache", "sir dard": "headache",
    "bukhar": "high_fever", "बुखार": "high_fever", "tap": "mild_fever", "fever": "high_fever",
    "pet dard": "stomach_pain", "पेट दर्द": "stomach_pain", "stomach pain": "stomach_pain",
    "khasi": "cough", "खांसी": "cough", "khokla": "cough", "cough": "cough",
    "thakan": "fatigue", "थकान": "fatigue", "thakwa": "fatigue", "fatigue": "fatigue",
    "ulti": "vomiting", "उल्टी": "vomiting", "vomit": "vomiting",
    "dast": "diarrhoea", "दस्त": "diarrhoea", "loose motion": "diarrhoea", "लेटरींग": "diarrhoea",
    "chills": "chills", "thand": "chills", "ठंड": "chills",
    "khujli": "itching", "खुजली": "itching", "itching": "itching"
}

# --- 2. Expanded Generic Medicine Knowledge Base (25+ Most Used Indian Formulations) ---
MEDICINE_DATABASE = {
    "paracetamol": {
        "name": "Paracetamol (Dolo 650 / Paracip 500 / Calpol)",
        "aliases": ["paracip", "dolo", "calpol", "crocin", "acetaminophen", "pacimol", "febrex", "wracip", "poeromo", "tablets ip 500", "500 mg", "650 mg"],
        "generic_info": "Generic Salt: Paracetamol IP (500mg / 650mg). Available at all PM Jan Aushadhi Kendras.",
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
    "pantoprazole_domperidone": {
        "name": "Pantoprazole + Domperidone (Pan-D / Pantocid-D)",
        "aliases": ["pan-d", "pand", "pantocid-d", "pantosec-d", "pantodac-dsr"],
        "generic_info": "Generic Salt: Pantoprazole (40mg) + Domperidone (30mg SR) Capsule.",
        "composition": "Acid Reducer + Prokinetic Anti-emetic",
        "usage": "Acidity accompanied by nausea, morning vomiting, indigestion, and acid fullness.",
        "safety_note": "Strictly take empty stomach in the morning with plain water."
    },
    "combiflam": {
        "name": "Ibuprofen + Paracetamol (Combiflam / Flexon)",
        "aliases": ["combiflam", "flexon", "brufen", "ibugesic plus", "ibuprofen paracetamol"],
        "generic_info": "Generic Salt: Ibuprofen (400mg) + Paracetamol (325mg) Tablet IP.",
        "composition": "Dual-action NSAID Analgesic & Anti-inflammatory",
        "usage": "Acute muscular pain, sprains, dental surgery pain, and joint swelling.",
        "safety_note": "Always consume strictly after a full meal to prevent stomach gastric irritation."
    },
    "metformin": {
        "name": "Metformin 500mg (Glycomet 500 / Obimet)",
        "aliases": ["glycomet", "obimet", "metfor", "metformin 500", "metformin hydrochloride"],
        "generic_info": "Generic Salt: Metformin Hydrochloride Sustained Release IP 500mg.",
        "composition": "Biguanide Class Antidiabetic Agent",
        "usage": "Controls blood sugar levels in Type 2 Diabetes Mellitus and PCOS.",
        "safety_note": "Take with or after main meals to avoid stomach upset. Monitor sugar regularly."
    },
    "telmisartan": {
        "name": "Telmisartan 40mg (Telma 40 / Telvas 40)",
        "aliases": ["telma", "telvas", "telmikind", "telsartan", "telmisartan 40"],
        "generic_info": "Generic Salt: Telmisartan Tablets IP 40mg.",
        "composition": "Angiotensin II Receptor Blocker (Antihypertensive)",
        "usage": "Lowers high blood pressure (hypertension) and protects heart/kidneys.",
        "safety_note": "Take at the same fixed time daily. Do not stop abruptly without doctor advice."
    },
    "ors": {
        "name": "Oral Rehydration Salts (WHO-Formula ORS / Electral)",
        "aliases": ["electral", "ors", "rehydrate", "w.h.o. formula", "energy drink powder"],
        "generic_info": "Generic Salt: Standard WHO Formulation Oral Rehydration Salts.",
        "composition": "Sodium Chloride + Potassium Chloride + Sodium Citrate + Anhydrous Dextrose",
        "usage": "Treats severe dehydration caused by diarrhoea, vomiting, summer heat, or heavy sweating.",
        "safety_note": "Dissolve entire packet in correct measured water (usually 1 litre). Consume within 24 hours."
    }
}

# --- 3. NLP Symptom Processing Layer ---
def process_nlp_symptoms(raw_symptoms_list):
    final_symptoms = []
    combined_raw = " ".join([str(s) for s in raw_symptoms_list]).lower()
    
    for phrase, mapped_key in LOCAL_TERM_MAP.items():
        if phrase in combined_raw:
            final_symptoms.append(mapped_key)
            
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(combined_raw).lower()
    except:
        translated = combined_raw
        
    for symp in dataset_symptoms:
        clean_name = symp.replace('_', ' ').lower()
        if clean_name in translated or clean_name in combined_raw:
            final_symptoms.append(symp)
            
    words = translated.split()
    for word in words:
        if len(word) >= 4 and dataset_symptoms:
            match, score = process.extractOne(word, dataset_symptoms)
            if score > 75:
                final_symptoms.append(match)
                
    return list(set(final_symptoms))

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
            return jsonify({'error': 'Kripya kam se kam ek symptom likhein ya select karein.'})
            
        smart_symptoms = process_nlp_symptoms(frontend_symptoms)
        
        if not smart_symptoms:
            return jsonify({'error': 'System aapke lakshan pehchan nahi paya. Kripya aam terms likhein (jaise: bukhar, sir dard, ulti, khasi).'})
            
        # --- FEATURE 1: CLINICAL SAFEGUARD & FOLLOW-UP ON SINGLE SYMPTOM ---
        if len(smart_symptoms) == 1:
            single = smart_symptoms[0]
            if single in ['high_fever', 'mild_fever']:
                return jsonify({
                    'disease': 'Viral Pyrexia (Acute Viral Fever)',
                    'is_single': True,
                    'identified_symptom': 'Fever / Bukhar',
                    'follow_up_question': 'Aapko bukhar ke sath inme se aur kya mehsoos ho raha hai?',
                    'follow_up_options': ['Thand lagna (Chills)', 'Sir dard (Headache)', 'Ulti (Vomiting)', 'Body pain / Thakan'],
                    'description': 'Sirf bukhar aana aam viral infection ya seasonal badlav ka sanket hai. Agar bukhar 3 din se zyada rahe toh clinical test (CBC/Widal) zaroori hai.',
                    'precautions': ['Paracetamol (500mg) as advised for temperature', 'Khoob sara paani aur ORS/Nariyal paani piyein', 'Gili patti (Cold compress) lagayein agar bukhar tez ho', '3 din se zyada bukhar par doctor se checkup karwayein']
                })
            elif single in ['headache']:
                return jsonify({
                    'disease': 'Tension Headache / Migraine Cephalea',
                    'is_single': True,
                    'identified_symptom': 'Headache / Sir Dard',
                    'follow_up_question': 'Sir dard ke sath koi aur pareshani hai?',
                    'follow_up_options': ['Ulti / Nausea', 'Aankhon me jalan', 'Chakkar aana', 'Gardan me dard'],
                    'description': 'Akela sir dard aam taur par thakan, dehydration, screen stress ya neend ki kami se hota hai.',
                    'precautions': ['Shant aur andhere kamre me aaram karein', 'Pani ki matra badhayein (Hydration)', 'Mobile/Laptop screen time kam karein', 'Agar ulti ke sath ho toh doctor ko dikhayein']
                })
            elif single in ['cough']:
                return jsonify({
                    'disease': 'Upper Respiratory Tract Irritation (Common Cough)',
                    'is_single': True,
                    'identified_symptom': 'Cough / Khasi',
                    'follow_up_question': 'Khasi ke sath koi aur lakshan hai?',
                    'follow_up_options': ['Gale me kharash', 'Bukhar (Fever)', 'Balgum / Phlegm', 'Chheenkein (Sneezing)'],
                    'description': 'Sookhi ya balgam wali khasi aam viral infection ya dust allergy se hoti hai.',
                    'precautions': ['Gungune paani me namak daal kar gargle karein', 'Steam inhalation (Bhaap) lein', 'Thandi aur tali cheezon se parhez karein', 'Adrak-tulsi chai ya honey lein']
                })

        # Multi-symptom processing with Random Forest
        input_features = [0] * len(dataset_symptoms)
        for symptom in smart_symptoms:
            if symptom in dataset_symptoms:
                idx = dataset_symptoms.index(symptom)
                input_features[idx] = 1
                
        features_array = np.array([input_features])
        prediction = model.predict(features_array)[0]
        disease_name = str(prediction).strip()
        
        disease_desc = f"Clinical pattern matched {len(smart_symptoms)} symptom(s): {', '.join([s.replace('_',' ').title() for s in smart_symptoms])}."
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

# --- FEATURE 3: EXPANDED MEDICINE SCANNER MATCHER ---
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
                'generic_info': 'Generic Salt Equivalent: Verify the salt name on packaging with your local pharmacist.',
                'composition': f"Extracted Text: {raw_text[:120].strip()}",
                'usage': 'Prescription medication detected. Follow doctor/pharmacist dosage directions.',
                'safety_note': 'Ensure batch number and expiry date are verified before consuming.'
            })
            
    except Exception as e:
        return jsonify({'error': f'Scanner processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)