import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration (Website ko wide aur professional banane ke liye)
st.set_page_config(page_title="MediSmart AI", page_icon="🩺", layout="wide")

# 2. Data Loading with Cache (Taaki website bar-bar load hone par fast chale)
@st.cache_resource
def load_data():
    model = pickle.load(open('model.pkl', 'rb'))
    symptoms_list = pickle.load(open('columns.pkl', 'rb'))
    description_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
    return model, symptoms_list, description_df, precaution_df

model, symptoms_list, description_df, precaution_df = load_data()

# 3. Sidebar Setup (Professional look ke liye left menu)
st.sidebar.title("🩺 MediSmart AI")
st.sidebar.markdown("---")
st.sidebar.info("Welcome to MediSmart! Select your symptoms from the dropdown and our AI will predict the most likely disease and provide basic precautions.")
st.sidebar.warning("⚠️ **Disclaimer:** This is an AI project for educational purposes only. Always consult a certified doctor for medical advice.")
st.sidebar.markdown("---")
st.sidebar.text("Developed by: B.Tech Data Science Student")

# 4. Main UI Design
st.title("👨‍⚕️ MediSmart: Intelligent Symptom Checker")
st.markdown("Enter your symptoms below to get an AI-driven preliminary diagnosis.")

# Input Section
st.markdown("### 📝 Patient Symptoms")
selected_symptoms = st.multiselect("Search and select your symptoms:", symptoms_list)

# Primary Button
if st.button("🔍 Predict Disease", type="primary"):
    if len(selected_symptoms) > 0:
        # Prediction Logic
        input_data = np.zeros(len(symptoms_list))
        for symptom in selected_symptoms:
            index = symptoms_list.index(symptom)
            input_data[index] = 1
            
        prediction = model.predict([input_data])[0]
        
        # UI Divider
        st.markdown("---")
        st.subheader("📊 Diagnosis Results")
        
        # Bada Highlighted Prediction Box
        st.success(f"### 🛑 Predicted Disease: **{prediction.upper()}**")
        
        # 5. Columns Layout (Description aur Precautions ko aamne-saamne dikhane ke liye)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📖 About the Disease")
            try:
                # Column ka naam kuch bhi ho, hum dynamically second column utha lenge (error fix)
                desc_col_name = description_df.columns[1] 
                desc = description_df[description_df['Disease'] == prediction][desc_col_name].values
                if len(desc) > 0:
                    st.info(desc[0])
                else:
                    st.write("Description data is not available for this disease.")
            except Exception as e:
                st.error("Error loading description.")

        with col2:
            st.markdown("#### 🛡️ Recommended Precautions (Dos & Don'ts)")
            try:
                precautions = precaution_df[precaution_df['Disease'] == prediction].values
                if len(precautions) > 0:
                    # Precautions list nikalna aur cleanly dikhana
                    for i in range(1, 5):
                        if pd.notna(precautions[0][i]):
                            st.warning(f"✅ {str(precautions[0][i]).title()}")
            except:
                 st.write("Precautions data is not available.")
    else:
        st.error("Kripya diagnosis ke liye kam se kam ek symptom select karein!")