# MediSmart AI — Clinical Intelligence & Triage Suite

MediSmart is a hybrid clinical triage and medicine assistance web platform combining **Google Gemini LLM**, **Random Forest Machine Learning**, and **Client-Side OCR**.

---

## Key Features

- **Conversational AI Triage:** Multilingual natural language symptom extraction powered by Google Gemini (supports Hindi, Hinglish, Marathi, and English).
- **Clinical Safety Guard:** Prevents false-positive severe disease predictions on isolated/vague symptoms using clinical rule-based validation.
- **Manual Symptom Checklist:** 130+ clinical taxonomy selector backed by a Random Forest ML classifier.
- **Medicine Strip Scanner:** Dual-scan OCR engine identifying brand names, active salts, and PM Jan Aushadhi generic alternatives.

---

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **AI / ML:** Google Generative AI (Gemini 1.5 Flash), Scikit-Learn, Pandas, NumPy
- **NLP & Matching:** FuzzyWuzzy, Deep-Translator
- **Frontend:** HTML5, CSS3 (Royal Indigo Theme), JavaScript, Select2, Tesseract.js

---

## Project Structure

```text
MediSmart_Project/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── model.pkl
├── columns.pkl
├── symptom_Description.csv
├── symptom_precaution.csv
└── templates/
    └── index.html