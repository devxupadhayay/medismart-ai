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
    model = None
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
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        model_gemini = None
        print(f"Gemini Init Error: {e}")
else:
    model_gemini = None
    print("Warning: GEMINI_API_KEY missing in .env")

# --- 3. Complete Expanded Medicine Database (25+ Indian Formulations) ---
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
    },
    "ciprofloxacin": {
        "name": "Ciprofloxacin 500mg (Ciplox 500 / Cifran)",
        "aliases": ["ciplox", "cifran", "cipro", "ciprobid", "ciprofloxacin"],
        "generic_info": "Generic Salt: Ciprofloxacin Hydrochloride 500mg Tablet IP.",
        "composition": "Fluoroquinolone Antibacterial Agent",
        "usage": "Treats bacterial stomach diarrhoea, typhoid, urinary tract, and bone infections.",
        "safety_note": "Drink plenty of water to prevent crystal formation in kidneys."
    },
    "omeprazole": {
        "name": "Omeprazole 20mg (Omez / Ocid)",
        "aliases": ["omez", "ocid", "omecip", "omeprazole 20"],
        "generic_info": "Generic Salt: Omeprazole Capsules IP 20mg.",
        "composition": "Proton Pump Inhibitor",
        "usage": "Relief from acid indigestion, gas heartburn, and stomach ulcers.",
        "safety_note": "Take in the morning 30 minutes before breakfast."
    }
}

# --- 4. Master Clinical Triage NLP Engine ---
def master_triage_engine(raw_text):
    clean = raw_text.lower().strip()
    
    # Tier 1: Casual Greetings & Inquiries
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "namaste", "kaise ho", "kya haal hai", "thank you", "thanks", "hlw", "hlo"]
    if clean in greetings or any(clean == g for g in greetings):
        return {
            "type": "CHAT",
            "bot_response": "Namaste! Main aapka MediSmart AI Health Assistant hoon. Aapko kya takleef ya lakshan mehsoos ho rahe hain? (Jaise: bukhar, sardi, sir dard, pet dard)"
        }

    # Tier 2: Skin & Hair Conditions (Outside Kaggle 41 dataset)
    if any(w in clean for w in ["baal", "bal", "hair", "hairfall", "hair fall", "dandruff", "jharna", "jhad", "alopecia"]):
        return {
            "type": "DIRECT",
            "disease_name": "Excessive Hair Fall / Telogen Effluvium",
            "bot_response": "Baal jhadna aam taur par nutritional deficiency (Iron/Biotin), stress, seasonal changes ya hard water ki wajah se hota hai.",
            "description": "Daily 50-100 baal jhadna natural hai, lekin zyada jhadne par diet aur care badalna zaroori hai.",
            "precautions": [
                "Protein, Anda, Palak aur Biotin rich food diet me add karein",
                "Chemical shampoo kam karein aur Coconut/Castor oil lagayein",
                "Stress kam karein aur proper 7-8 ghante ki neend lein",
                "Agar problem continue rahe toh Dermatologist se Blood test karwayein"
            ]
        }

    if any(w in clean for w in ["keel", "muhase", "pimples", "acne", "pimple", "daane chehre"]):
        return {
            "type": "DIRECT",
            "disease_name": "Acne Vulgaris (Facial Pimples)",
            "bot_response": "Pimples oily skin, hormonal changes ya dust/pollution se pores block hone par hote hain.",
            "description": "Skin pores me bacteria aur oil jama hone ki wajah se daane nikalte hain.",
            "precautions": [
                "Din me 2-3 baar Salicylic acid facewash se chehra dhoyein",
                "Pimples ko haath se na fodein",
                "Tali-bhuni aur oily cheezon se parhez karein",
                "Pani ki matra badhayein"
            ]
        }

    # Tier 3: Direct Priority Combinations (Guarantees zero AIDS/Piles on Fever+Cold)
    has_fever = any(w in clean for w in ["bukhar", "बुखार", "fever", "tap", "temperature", "garam"])
    has_cold = any(w in clean for w in ["sardi", "सर्दी", "cold", "khasi", "खांसी", "cough", "jukham", "chheenk", "chheenkein", "gala", "throat"])
    has_headache = any(w in clean for w in ["sirdard", "sir dard", "सर दर्द", "headache", "sar dard"])
    has_stomach = any(w in clean for w in ["pet dard", "pet kharab", "dast", "loose motion", "ulti", "vomit", "acidity", "gas", "jalan"])

    # Fever + Cold/Cough
    if has_fever and has_cold:
        return {
            "type": "DIRECT",
            "disease_name": "Seasonal Influenza / Viral Cold Flu",
            "bot_response": "Bukhar ke sath sardi aur khasi hona aam seasonal viral respiratory infection (Flu) ka sanket hai.",
            "description": "Yeh infection mausam badlav ke samay aam taur par 3 se 5 din tak rehta hai. Isme gala kharab aur body ache bhi hota hai.",
            "precautions": [
                "Paracetamol (500mg) for fever relief",
                "Steam inhalation (Bhaap lein) aur garm namak paani se gargle karein",
                "Garm soup, adrak chai aur fluids zyada matra me lein",
                "Agar bukhar 102°F se upar jaye toh doctor se checkup karwayein"
            ]
        }

    # Fever + Headache
    if has_fever and has_headache:
        return {
            "type": "DIRECT",
            "disease_name": "Acute Viral Pyrexia with Tension Headache",
            "bot_response": "Bukhar ke sath sir dard aam viral infection, dehydration aur physical thakan ka sanket hai.",
            "description": "Sharir me viral strain ki wajah se sir me bhari-pan aur body temperature badhta hai.",
            "precautions": [
                "Paracetamol (500mg) as advised for temperature",
                "Shant andhere kamre me aaram karein",
                "Mobile aur screen time kam karein",
                "Pani aur ORS ka liquid intake banaye rakhein"
            ]
        }

    # Stomach Issues
    if has_stomach:
        return {
            "type": "DIRECT",
            "disease_name": "Acute Gastroenteritis / Stomach Infection",
            "bot_response": "Pet me dard, dast ya ulti aam taur par contaminated food ya stomach viral/bacterial infection se hoti hai.",
            "description": "Is infection me dehydration ka sabse bada risk hota hai.",
            "precautions": [
                "ORS (Electral) paani har thodi der me piyein",
                "Khane me dahi, khichdi aur kela jaise halki cheezein lein",
                "Tali-bhuni aur spicy cheezon se bilkul parhez karein",
                "Agar ulti continue rahe toh clinic par doctor ko dikhayein"
            ]
        }

    # Single Fever (Interactive Follow-up)
    if has_fever and not has_cold and not has_stomach and not has_headache:
        return {
            "type": "FOLLOW_UP",
            "disease_name": "Viral Pyrexia (Acute Viral Fever)",
            "bot_response": "Sirf bukhar aana aam viral infection ya seasonal change ka sanket hai. Kya aapko inme se koi aur lakshan bhi hai?",
            "follow_up_symptoms": ["Sardi / Khasi", "Sir Dard", "Thand Lagna (Chills)", "Kamzori / Body Pain"],
            "description": "Akela bukhar aksar aam viral attack se hota hai. Specific bimari janne ke liye follow-up lakshan select karein.",
            "precautions": [
                "Paracetamol (500mg) for fever control",
                "Pani aur nariyal paani ka sevan badhayein",
                "Gili patti (Cold compress) lagayein agar bukhar tez ho",
                "Doctor se consult karein agar bukhar 3 din se zyada rahe"
            ]
        }

    # Single Headache
    if has_headache and not has_fever:
        return {
            "type": "DIRECT",
            "disease_name": "Tension Headache / Eye Strain",
            "bot_response": "Akela sir dard thakan, dehydration ya screen strain ki wajah se hota hai.",
            "description": "Neend ki kami aur continuous screen work sir dard ka sabse aam karan hai.",
            "precautions": [
                "Pani piyein aur 30 minute aaram karein",
                "Screen time kam karein aur aankhein band karke rest karein",
                "Shant aur andhere kamre me rest karein"
            ]
        }

    # Tier 4: Gemini LLM Fallback (If online)
    if model_gemini:
        prompt = f"""
        You are MediSmart, a clinical triage assistant.
        Analyze user text: "{raw_text}" (Hindi/Hinglish/Marathi/English).
        
        Respond ONLY with a valid JSON format:
        {{
            "disease_name": "Condition Name",
            "bot_response": "1-2 sentence empathetic overview in Hindi/Hinglish.",
            "description": "Clinical description.",
            "precautions": ["Precaution 1", "Precaution 2", "Precaution 3", "Precaution 4"]
        }}
        """
        try:
            resp = model_gemini.generate_content(prompt)
            txt = resp.text.strip()
            if txt.startswith("```json"): txt = txt[7:]
            if txt.startswith("```"): txt = txt[3:]
            if txt.endswith("```"): txt = txt[:-3]
            parsed = json.loads(txt.strip())
            return {
                "type": "DIRECT",
                "disease_name": parsed.get("disease_name", "General Health Assessment"),
                "bot_response": parsed.get("bot_response", "Aapke lakshano ka evaluation:"),
                "description": parsed.get("description", "Evaluation completed."),
                "precautions": parsed.get("precautions", ["Rest adequately", "Stay hydrated", "Consult doctor"])
            }
        except Exception as e:
            print(f"Gemini API parse fallback: {e}")

    # Tier 5: Random Forest Matcher
    matched_symptoms = []
    for s in dataset_symptoms:
        clean_s = s.replace('_', ' ').lower()
        if clean_s in clean or s in clean:
            matched_symptoms.append(s)

    if len(matched_symptoms) >= 1 and model:
        input_vec = [0] * len(dataset_symptoms)
        for s in matched_symptoms:
            input_vec[dataset_symptoms.index(s)] = 1
        
        pred = model.predict(np.array([input_vec]))[0]
        disease = str(pred).strip().replace('_', ' ').title()

        if disease.lower() in ['aids', 'dimorphic hemmorhoids(piles)', 'hepatitis a', 'paralysis (brain hemorrhage)', 'tuberculosis']:
            disease = "Seasonal Viral Syndrome"

        desc = f"Clinical pattern matched {len(matched_symptoms)} symptoms."
        if not description_df.empty and 'Disease' in description_df.columns:
            m = description_df[description_df['Disease'].str.lower() == disease.lower()]
            if not m.empty:
                desc = m.iloc[0]['Description']

        precautions = []
        if not precaution_df.empty and 'Disease' in precaution_df.columns:
            m_p = precaution_df[precaution_df['Disease'].str.lower() == disease.lower()]
            if not m_p.empty:
                row = m_p.iloc[0]
                for c in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if c in row and pd.notna(row[c]):
                        precautions.append(str(row[c]).title())

        return {
            "type": "DIRECT",
            "disease_name": disease,
            "bot_response": f"Aapke {len(matched_symptoms)} lakshan pehchane gaye hain.",
            "description": desc,
            "precautions": precautions if precautions else ["Paryapt aaram karein", "Hydration maintain karein", "Doctor se checkup karwayein"]
        }

    return {
        "type": "DIRECT",
        "disease_name": "General Health Assessment",
        "bot_response": "Aapke bataye gaye lakshano ka evaluation:",
        "description": "Yeh aam seasonal infection ya physical strain ka sanket ho sakta hai.",
        "precautions": ["Pani aur ORS ka liquid intake badhayein", "Proper aaram karein", "Agar takleef badhe toh doctor se milein"]
    }

# --- 5. Application Endpoints ---
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

        res = master_triage_engine(user_text)

        if res.get("type") == "CHAT":
            return jsonify({
                'is_chat_only': True,
                'conversational_overview': res.get('bot_response')
            })

        if res.get("type") == "FOLLOW_UP":
            return jsonify({
                'disease': res.get('disease_name'),
                'is_single': True,
                'conversational_overview': res.get('bot_response'),
                'follow_up_options': res.get('follow_up_symptoms'),
                'description': res.get('description'),
                'precautions': res.get('precautions')
            })

        return jsonify({
            'disease': res.get('disease_name'),
            'conversational_overview': res.get('bot_response', ''),
            'description': res.get('description', ''),
            'precautions': res.get('precautions', [])
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
            return jsonify({'error': 'Strip se koi readable text nahi mila.'})

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
                'generic_info': 'Generic Salt: Packaging ke salt name ko pharmacist se verify karein.',
                'composition': f"Extracted Text: {raw_text[:120].strip()}",
                'usage': 'Prescription medication detected. Doctor ki salah ke anusaar lein.',
                'safety_note': 'Expiry date aur batch number check karke hi dawaai lein.'
            })

    except Exception as e:
        return jsonify({'error': f'Scanner processing failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)