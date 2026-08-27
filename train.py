import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

print("1. Dataset load ho raha hai...")
# Dataset read karein
df = pd.read_csv('dataset.csv')

# Bimari ka naam (Disease) hata kar baaki columns (symptoms) ki list banayein
symptom_columns = [col for col in df.columns if col != 'Disease']

print("2. Data ko Machine Learning ke liye prepare kar rahe hain...")
# Saare unique symptoms ko ek list mein nikalna aur extra spaces hatana
all_symptoms = pd.unique(df[symptom_columns].values.ravel('K'))
all_symptoms = [symp.strip() for symp in all_symptoms if pd.notna(symp)]
unique_symptoms = list(set(all_symptoms))

# 0 aur 1 ka naya dataset banana (0 = Symptom nahi hai, 1 = Symptom hai)
X = pd.DataFrame(0, index=np.arange(len(df)), columns=unique_symptoms)
y = df['Disease']

# Har bimari ke saamne uske symptoms ko 1 mark karna
for i in range(len(df)):
    for col in symptom_columns:
        symp = df.iloc[i][col]
        if pd.notna(symp):
            X.loc[i, symp.strip()] = 1

print("3. Model train ho raha hai, thoda wait karein...")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("4. Model aur Symptoms ki list save ho rahi hai...")
# Trained model ko save karna
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Symptoms ki list ko save karna taaki website mein use kar sakein
with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("✅ Success! 'model.pkl' aur 'columns.pkl' successfully ban gayi hain.")