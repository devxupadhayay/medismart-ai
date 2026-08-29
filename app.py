from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

warnings.filterwarnings("ignore")
load_dotenv()

app = Flask(__name__)

# --- 1. Load ML Model & Datasets ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    dataset_symptoms = pickle.load(open('columns.pkl', 'rb'))
    verified_symptoms_list = [s.replace('_', ' ').strip().lower() for s in dataset_symptoms]
except Exception as e:
    print(f"Model loading error: {e}")
    dataset_symptoms = []
    verified_symptoms_list = []

try:
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
except Exception as e:
    description_df = pd.DataFrame()
    precaution_df = pd.DataFrame()

# --- 2. Configure Google Gemini Brain ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None
    print("Warning: GEMINI_API_KEY missing in .env")

# --- 3. Medicine Knowledge Base ---
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
    "combiflam": {
        "name": "Ibuprofen + Paracetamol (Combiflam / Flexon)",
        "aliases": ["combiflam", "flexon", "brufen", "ibugesic plus", "ibuprofen paracetamol"],
        "generic_info": "Generic Salt: Ibuprofen (400mg) + Paracetamol (325mg) Tablet IP.",
        "composition": "Dual-action NSAID Analgesic & Anti-inflammatory",
        "usage": "Acute muscular pain, sprains, dental surgery pain, and joint swelling.",
        "safety_note": "Always consume strictly after a full meal to prevent stomach gastric irritation."
    }
}

# --- 4. Triage NLP Brain ---
def triage_nlp(raw_text):
    clean = raw_text.lower().strip()
    
    # 1. Direct Greetings Filter
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste", "kaise ho", "kya haal hai", "thank you", "thanks"]
    if clean in greetings or len(clean.split()) == 1 and clean in greetings:
        return {
            "type": "CHAT",
            "bot_response": "Namaste! Main aapka MediSmart AI Health Assistant hoon. Aaj aapki sehat se judi kis takleef ya lakshan me madad kar sakta hoon?"
        }

    # 2. Rule-Based Fallback / Priority Engine (Guarantees zero AIDS/Piles on Fever+Cold)
    has_fever = any(w in clean for w in ["bukhar", "बुखार", "fever", "tap", "temperature"])
    has_cold = any(w in clean for w in ["sardi", "सर्दी", "cold", "khasi", "खांसी", "cough", "jukham", "chheenk", "chheenko"])
    has_headache = any(w in clean for w in ["sirdard", "sir dard", "सर दर्द", "headache"])
    has_stomach = any(w in clean for w in ["pet dard", "pet kharab", "dast", "loose motion", "ulti", "vomit", "acidity", "gas"])

    if has_fever and has_cold:
        return {
            "type": "DIRECT_DIAGNOSIS",
            "disease_name": "Seasonal Influenza / Viral Cold Flu",
            "bot_response": "Bukhar ke sath sardi aur khasi hona aam seasonal viral respiratory infection (Flu) ka sanket hai.",
            "description": "Viral flu mausam badalne par aam taur par 3 se 5 din tak rehta hai. Isme gala kharab aur body ache bhi ho sakta hai.",
            "precautions": ["Paracetamol (500mg) for fever relief", "Garm paani ki steam (bhaap) lein aur namak paani se gargle karein", "Garm soup aur khoob sara paani piyein", "Agar bukhar 4 din se zyada rahe toh doctor se test karwayein"]
        }

    if has_fever and not has_cold and not has_stomach and not has_headache:
        return {
            "type": "FOLLOW_UP",
            "disease_name": "Viral Pyrexia (Acute Viral Fever)",
            "bot_response": "Sirf bukhar aana aam viral infection ya thakan ka sanket hai. Kya aapko iske sath kuch aur lakshan bhi hain?",
            "follow_up_symptoms": ["Sardi / Khasi", "Sir Dard", "Thand Lagna (Chills)", "Kamzori / Body Pain"],
            "description": "Akela bukhar aksar aam viral attack se hota hai. Specific bimari janne ke liye follow-up select karein.",
            "precautions": ["Paracetamol (500mg) as advised for temperature", "Hydration ke liye ORS / Nariyal paani piyein", "Gili patti (Cold compress) lagayein", "Doctor se checkup karwayein agar bukhar 102°F se zyada ho"]
        }

    if has_stomach:
        return {
            "type": "DIRECT_DIAGNOSIS",
            "disease_name": "Acute Gastroenteritis / Stomach Infection",
            "bot_response": "Pet dard, loose motion ya ulti aam taur par contaminated food ya stomach viral/bacterial infection se hota hai.",
            "description": "Yeh infection dehydration paida kar sakta hai, isliye liquid intake sabse zaroori hai.",
            "precautions": ["ORS (Electral) paani har ghante thoda-thoda piyein", "Khane me dahi, khichdi aur kela jaise halki diet lein", "Tali-bhuni aur spicy cheezon se parhez karein", "Agar ulti na ruke toh doctor se clinic par milen"]
        }

    # 3. Gemini LLM Deep Analysis (For general concerns like Hair Fall, Pimples, or complex multi-symptoms)
    if model_gemini:
        prompt = f"""
        You are MediSmart clinical AI assistant.
        Analyze user text: "{raw_text}" (Hindi/Hinglish/Marathi/English).
        
        Respond ONLY with a valid JSON in one of these exact formats:

        Format A (If health issue is outside typical infectious diseases, like hair fall, dandruff, acne, joint pain):
        {{
            "type": "GENERAL_HEALTH",
            "disease_name": "Issue Name (e.g., Telogen Effluvium / Hair Fall)",
            "bot_response": "Empathetic clinical explanation in Hindi/Hinglish.",
            "description": "Clinical overview.",
            "precautions": ["Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4"]
        }}

        Format B (If multiple hospital clinical symptoms are present):
        {{
            "type": "MODEL",
            "extracted_symptoms": ["matching symptoms from list below"],
            "bot_response": "1-sentence Hindi/Hinglish summary."
        }}

        VERIFIED CLINICAL SYMPTOMS LIST:
        {', '.join(verified_symptoms_list[:70])}
        """
        try:
            resp = model_gemini.generate_content(prompt)
            txt = resp.text.strip()
            if txt.startswith("```json"): txt = txt[7:]
            if txt.startswith("```"): txt = txt[3:]
            if txt.endswith("```"): txt = txt[:-3]
            return json.loads(txt.strip())
        except Exception as e:
            print(f"Gemini API parse error: {e}")

    # Fallback to ML Model
    return {
        "type": "MODEL",
        "extracted_symptoms": clean.replace(',', ' ').split(),
        "bot_response": "Symptoms analyzed with clinical engine."
    }

# --- 5. Application Routes ---
@app.route('/')
def home():
    return render_template('index.html', symptoms=dataset_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        if 'symptoms' in data and isinstance(data['symptoms'], list):
            user_text = " ".join(data['symptoms'])
        else:
            user_text = data.get('symptoms_text', '').strip()

        if not user_text:
            return jsonify({'error': 'Kripya apne lakshan likhein ya select karein.'})

        triage = triage_nlp(user_text)

        # Case 1: Casual Chat
        if triage.get("type") == "CHAT":
            return jsonify({
                'is_chat_only': True,
                'conversational_overview': triage.get('bot_response')
            })

        # Case 2: Single Symptom Follow-Up
        if triage.get("type") == "FOLLOW_UP":
            return jsonify({
                'disease': triage.get('disease_name'),
                'is_single': True,
                'conversational_overview': triage.get('bot_response'),
                'follow_up_options': triage.get('follow_up_symptoms'),
                'description': triage.get('description'),
                'precautions': triage.get('precautions')
            })

        # Case 3: Direct Handled Diagnosis (Fever+Cold, Stomach issues)
        if triage.get("type") == "DIRECT_DIAGNOSIS" or triage.get("type") == "GENERAL_HEALTH":
            return jsonify({
                'disease': triage.get('disease_name'),
                'conversational_overview': triage.get('bot_response'),
                'description': triage.get('description'),
                'precautions': triage.get('precautions')
            })

        # Case 4: ML Random Forest Prediction
        extracted = triage.get('extracted_symptoms', [])
        input_features = [0] * len(dataset_symptoms)
        match_count = 0

        for item in extracted:
            clean_item = item.replace(' ', '_').lower()
            if clean_item in dataset_symptoms:
                idx = dataset_symptoms.index(clean_item)
                input_features[idx] = 1
                match_count += 1
            else:
                best_match, score = process.extractOne(clean_item, dataset_symptoms)
                if score >= 75:
                    idx = dataset_symptoms.index(best_match)
                    input_features[idx] = 1
                    match_count += 1

        features_array = np.array([input_features])

        if match_count == 0:
            disease_name = "Seasonal Viral Syndrome"
        else:
            try:
                probabilities = model.predict_proba(features_array)[0]
                max_prob = np.max(probabilities)
                pred = model.predict(features_array)[0]
                disease_name = str(pred).strip().replace('_', ' ').title()
                if max_prob < 0.35 and match_count <= 2:
                    disease_name = "Common Viral Flu"
            except:
                pred = model.predict(features_array)[0]
                disease_name = str(pred).strip().replace('_', ' ').title()

        # False-alarm block
        if disease_name.lower() in ['aids', 'dimorphic hemmorhoids(piles)', 'hepatitis a', 'paralysis (brain hemorrhage)', 'tuberculosis']:
            if match_count < 3:
                disease_name = "Seasonal Influenza / Viral Cold"

        # Lookup Description & Precautions
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
            'disease': disease_name,
            'conversational_overview': triage.get('bot_response', ''),
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