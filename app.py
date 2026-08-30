from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
from fuzzywuzzy import fuzz, process

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. Load ML Model & Reference Datasets ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    dataset_symptoms = pickle.load(open('columns.pkl', 'rb'))
except Exception as e:
    model = None
    dataset_symptoms = []

try:
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
except Exception as e:
    description_df = pd.DataFrame()
    precaution_df = pd.DataFrame()

# --- 2. Master Multilingual Intent & Symptom Lexicon ---
GREETINGS = [
    "good morning", "good evening", "good afternoon", "good night", "hello", "hi", "hey",
    "namaste", "नमस्ते", "नमस्कार", "kaise ho", "kya haal", "hlw", "hlo", "thank you", "thanks"
]

HAIR_TERMS = [
    "बाल", "झड़", "झड़", "hair", "hairfall", "hair fall", "dandruff", "alopecia", "baal", "jharna", "jhad", "tute", "scalp"
]

ACNE_TERMS = [
    "pimple", "pimples", "acne", "कील", "मुहासे", "मुहाँसे", "daane", "chehre", "phusi"
]

FEVER_TERMS = [
    "बुखार", "bukhar", "fever", "tap", "ताप", "temperature", "garam", "pyrexia", "hararat", "हल्का बुखार", "तेज बुखार"
]

COLD_TERMS = [
    "सर्दी", "sardi", "cold", "khasi", "खांसी", "cough", "jukham", "जुकाम", "chheenk", "छींक", "gala", "गले", "kharash", "throat", "runny"
]

HEADACHE_TERMS = [
    "sir dard", "sirdard", "sir me dard", "सर दर्द", "सिर दर्द", "सरदर्द", "headache", "sar dard", "matha", "migraine"
]

STOMACH_TERMS = [
    "pet dard", "pet me dard", "पेट दर्द", "पेट में दर्द", "stomach pain", "abdominal pain",
    "dast", "दस्त", "loose motion", "latrine", "diarrhea", "diarrhoea",
    "ulti", "उल्टी", "vomit", "vomiting", "nausea", "ji michlana",
    "acidity", "gas", "jalan", "seene me jalan", "एसिडिटी", "गैस"
]

# --- 3. Medicine Knowledge Base ---
MEDICINE_DATABASE = {
    "paracetamol": {
        "name": "Paracetamol (Dolo 650 / Paracip 500 / Calpol)",
        "aliases": ["paracip", "dolo", "calpol", "crocin", "acetaminophen", "pacimol", "febrex", "wracip", "500 mg", "650 mg"],
        "generic_info": "Generic Salt: Paracetamol IP (500mg / 650mg). PM Jan Aushadhi Kendra available.",
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
    "pantoprazole": {
        "name": "Pantoprazole 40mg (Pan 40 / Pantocid / Pantodac)",
        "aliases": ["pan 40", "pan40", "pantocid", "pantosec", "pantodac", "pantoprazole", "pan-d", "pand"],
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

# --- 4. Triage NLP Processor ---
def parse_clinical_query(raw_text):
    text = raw_text.lower().strip()

    # 1. Greetings Intent
    if any(g == text or g in text for g in GREETINGS) and len(text.split()) <= 4:
        if not any(f in text for f in FEVER_TERMS + COLD_TERMS + STOMACH_TERMS):
            return {
                "type": "CHAT",
                "bot_response": "Namaste! Main aapka MediSmart AI Health Assistant hoon. Aaj aapki sehat se judi kis takleef ya lakshan me madad kar sakta hoon?"
            }

    # 2. Hair Fall Intent
    if any(p in text for p in HAIR_TERMS):
        return {
            "type": "DIRECT",
            "disease_name": "Excessive Hair Fall / Telogen Effluvium",
            "bot_response": "Baal jhadna aam taur par nutritional deficiency (Iron, Biotin, Protein), stress ya hard water ki wajah se hota hai.",
            "description": "Daily 50-100 baal jhadna natural hai, lekin zyada baal tutne par proper nutrition aur scalp care zaroori hai.",
            "precautions": [
                "Protein, Anda, Palak, Amla aur Biotin rich food diet me shamil karein",
                "Harsh chemical shampoo band karein aur mild sulphate-free shampoo use karein",
                "Coconut ya Castor oil se scalp par halke haath se massage karein",
                "Proper 7-8 ghante ki neend lein aur dermatologist se blood test (Ferritin/B12) karwayein"
            ]
        }

    # 3. Skin Acne Intent
    if any(p in text for p in ACNE_TERMS):
        return {
            "type": "DIRECT",
            "disease_name": "Acne Vulgaris (Facial Pimples)",
            "bot_response": "Pimples oily skin, hormonal imbalance ya pores block hone ki wajah se hote hain.",
            "description": "Sebum aur dead skin cells pores me jama hone se pimples aur redness paida hoti hai.",
            "precautions": [
                "Din me 2 baar Salicylic acid ya Neem facewash se chehra dhoyein",
                "Pimples ko haath se touch ya pop bilkul na karein",
                "Tali-bhuni, junk aur oily food se parhez karein",
                "Din me 3-4 litre paani piyein aur hydration banaye rakhein"
            ]
        }

    # Core Symptoms Flags
    is_fever = any(p in text for p in FEVER_TERMS)
    is_cold = any(p in text for p in COLD_TERMS)
    is_headache = any(p in text for p in HEADACHE_TERMS)
    is_stomach = any(p in text for p in STOMACH_TERMS)

    # 4. Fever + Cold (Flu)
    if is_fever and is_cold:
        return {
            "type": "DIRECT",
            "disease_name": "Seasonal Influenza / Viral Cold Flu",
            "bot_response": "Bukhar ke sath sardi, khasi aur gale me kharash aam viral flu infection ke lakshan hain.",
            "description": "Viral flu mausam badlav ke samay aam taur par 3 se 5 din tak rehta hai. Isme halka badan dard aur thakan bhi hoti hai.",
            "precautions": [
                "Paracetamol (500mg/650mg) as advised for fever and body ache",
                "Steam inhalation (Bhaap lein) din me 2 baar aur gungune namak paani se gargle karein",
                "Garm soup, adrak-tulsi chai aur liquid fluids zyada matra me lein",
                "Agar bukhar 102°F se zyada ho ya 4 din se upar rahe toh doctor se consult karein"
            ]
        }

    # 5. Fever + Headache
    if is_fever and is_headache:
        return {
            "type": "DIRECT",
            "disease_name": "Acute Viral Pyrexia with Tension Cephalea",
            "bot_response": "Bukhar ke sath sir dard viral infection, dehydration ya sharirik thakan ka sanket hai.",
            "description": "Sharir me viral strain aur dehydration ki wajah se sir me dard aur temperature badhta hai.",
            "precautions": [
                "Paracetamol (500mg) for fever and pain relief",
                "Shant aur andhere kamre me aaram karein",
                "Mobile, TV aur laptop screen time se parhez karein",
                "ORS ya nariyal paani piyein taaki dehydration na ho"
            ]
        }

    # 6. Stomach Infection
    if is_stomach:
        return {
            "type": "DIRECT",
            "disease_name": "Acute Gastroenteritis (Stomach Infection)",
            "bot_response": "Pet dard, loose motion ya ulti aam taur par food infection ya stomach virus ki wajah se hoti hai.",
            "description": "Is infection me dehydration ka sabse bada khatra hota hai, isliye liquid balance sabse zaroori hai.",
            "precautions": [
                "ORS (Electral) paani har 1 ghante me thoda-thoda lagatar piyein",
                "Khane me dahi, khichdi aur kela jaise light digestible food lein",
                "Tali-bhuni, masaledar aur bahar ki cheezon se bilkul parhez karein",
                "Agar ulti 24 ghante se zyada na ruke toh clinic par doctor se milen"
            ]
        }

    # 7. Isolated Fever (Follow-up chips)
    if is_fever and not is_cold and not is_headache and not is_stomach:
        return {
            "type": "FOLLOW_UP",
            "disease_name": "Viral Pyrexia (Acute Viral Fever)",
            "bot_response": "Sirf bukhar aana aam viral infection ya seasonal badlav ka sanket hai. Kya aapko inme se koi aur lakshan bhi hai?",
            "follow_up_symptoms": ["Sardi / Khasi", "Sir Dard", "Thand Lagna (Chills)", "Kamzori / Body Pain"],
            "description": "Akela bukhar aksar aam viral attack se hota hai. Specific bimari janne ke liye follow-up select karein.",
            "precautions": [
                "Paracetamol (500mg) as advised for temperature control",
                "Pani aur nariyal paani ka sevan badhayein",
                "Gili patti (Cold compress) lagayein agar temperature tez ho",
                "3 din se zyada bukhar par doctor se blood test karwayein"
            ]
        }

    # 8. Isolated Headache
    if is_headache and not is_fever:
        return {
            "type": "DIRECT",
            "disease_name": "Tension Headache / Screen Eye Strain",
            "bot_response": "Akela sir dard neend ki kami, screen stress ya dehydration ki wajah se hota hai.",
            "description": "Aam taur par continuous work aur dehydration se sir ki blood vessels me tension badhti hai.",
            "precautions": [
                "Khoob saara paani piyein aur 30 minute aaram karein",
                "Screen time band karein aur aankhein band karke rest karein",
                "Shant aur thande kamre me rest karein"
            ]
        }

    # 9. Dataset Machine Learning Matching
    matched_symptoms = []
    for s in dataset_symptoms:
        clean_s = s.replace('_', ' ').lower()
        if clean_s in text or s in text:
            matched_symptoms.append(s)

    if len(matched_symptoms) >= 1 and model:
        input_vec = [0] * len(dataset_symptoms)
        for s in matched_symptoms:
            input_vec[dataset_symptoms.index(s)] = 1
        
        pred = model.predict(np.array([input_vec]))[0]
        disease = str(pred).strip().replace('_', ' ').title()

        if disease.lower() in ['aids', 'dimorphic hemmorhoids(piles)', 'hepatitis a', 'paralysis (brain hemorrhage)', 'tuberculosis']:
            disease = "Seasonal Viral Syndrome"

        desc = f"Clinical pattern matched {len(matched_symptoms)} verified symptom(s)."
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

    # Default Contextual Summary
    return {
        "type": "DIRECT",
        "disease_name": "General Health Assessment",
        "bot_response": "Aapke bataye gaye lakshano ka assessment:",
        "description": "Yeh sharirik thakan, mausam badlav ya mild seasonal infection ka sanket ho sakta hai.",
        "precautions": [
            "Pani aur ORS ka liquid intake badhayein",
            "Proper 7-8 ghante ki neend aur aaram karein",
            "Agar takleef 3 din se zyada rahe toh clinic par doctor se milein"
        ]
    }

# --- 5. Flask Endpoints ---
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

        res = parse_clinical_query(user_text)

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
        return jsonify({'error': 'Diagnosis process karne me dikkat aayi. Kripya dobara try karein.'})

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