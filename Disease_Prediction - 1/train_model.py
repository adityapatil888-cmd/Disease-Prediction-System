import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# DISEASE DATA
# ============================================================

data = [

    # Common Cold
    ["fever", "cough", "sore_throat", "runny_nose", "headache", "Common Cold"],
    ["cough", "runny_nose", "sneezing", "sore_throat", "Common Cold"],
    ["mild_fever", "cough", "sneezing", "runny_nose", "Common Cold"],
    ["headache", "cough", "runny_nose", "sore_throat", "Common Cold"],
    ["cough", "sneezing", "runny_nose", "fatigue", "Common Cold"],

    # Flu
    ["high_fever", "cough", "headache", "body_pain", "fatigue", "Flu"],
    ["fever", "body_pain", "chills", "headache", "fatigue", "Flu"],
    ["high_fever", "cough", "sore_throat", "body_pain", "Flu"],
    ["fever", "headache", "chills", "weakness", "Flu"],
    ["cough", "fever", "body_pain", "fatigue", "Flu"],

    # Migraine
    ["headache", "nausea", "vomiting", "light_sensitivity", "Migraine"],
    ["severe_headache", "nausea", "light_sensitivity", "Migraine"],
    ["headache", "vomiting", "noise_sensitivity", "Migraine"],
    ["severe_headache", "nausea", "noise_sensitivity", "Migraine"],
    ["headache", "blurred_vision", "nausea", "Migraine"],

    # Gastritis
    ["stomach_pain", "nausea", "vomiting", "acidity", "Gastritis"],
    ["stomach_pain", "acidity", "indigestion", "Gastritis"],
    ["abdominal_pain", "nausea", "loss_of_appetite", "Gastritis"],
    ["acidity", "stomach_pain", "bloating", "Gastritis"],
    ["indigestion", "bloating", "stomach_pain", "Gastritis"],

    # Allergy
    ["sneezing", "itchy_eyes", "runny_nose", "skin_rash", "Allergy"],
    ["skin_rash", "itching", "redness", "Allergy"],
    ["sneezing", "itchy_eyes", "runny_nose", "Allergy"],
    ["itching", "skin_rash", "watery_eyes", "Allergy"],
    ["runny_nose", "sneezing", "itching", "Allergy"],

    # Diabetes
    ["frequent_urination", "excessive_thirst", "fatigue", "Diabetes"],
    ["excessive_thirst", "weight_loss", "frequent_urination", "Diabetes"],
    ["fatigue", "blurred_vision", "excessive_thirst", "Diabetes"],
    ["frequent_urination", "weight_loss", "fatigue", "Diabetes"],
    ["excessive_hunger", "excessive_thirst", "fatigue", "Diabetes"],

    # Hypertension
    ["headache", "dizziness", "blurred_vision", "Hypertension"],
    ["headache", "dizziness", "chest_discomfort", "Hypertension"],
    ["dizziness", "fatigue", "headache", "Hypertension"],
    ["blurred_vision", "headache", "dizziness", "Hypertension"],
    ["headache", "fatigue", "chest_discomfort", "Hypertension"],

    # Food Poisoning
    ["vomiting", "diarrhea", "stomach_pain", "nausea", "Food Poisoning"],
    ["diarrhea", "vomiting", "fever", "stomach_pain", "Food Poisoning"],
    ["nausea", "diarrhea", "abdominal_pain", "Food Poisoning"],
    ["vomiting", "diarrhea", "weakness", "Food Poisoning"],
    ["stomach_pain", "vomiting", "diarrhea", "Food Poisoning"],
]


# ============================================================
# GET ALL SYMPTOMS
# ============================================================

all_symptoms = set()

for row in data:
    for symptom in row[:-1]:
        all_symptoms.add(symptom)

all_symptoms = sorted(all_symptoms)


# ============================================================
# CREATE BINARY DATASET
# ============================================================

dataset = []

for row in data:

    symptoms = set(row[:-1])
    disease = row[-1]

    values = []

    for symptom in all_symptoms:

        if symptom in symptoms:
            values.append(1)
        else:
            values.append(0)

    values.append(disease)

    dataset.append(values)


columns = all_symptoms + ["disease"]

df = pd.DataFrame(dataset, columns=columns)


# ============================================================
# X AND Y
# ============================================================

X = df[all_symptoms]
y = df["disease"]


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# ACCURACY
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/disease_model.pkl"
)

joblib.dump(
    all_symptoms,
    "model/symptoms.pkl"
)


print()
print("=" * 55)
print("       DISEASE PREDICTION ML MODEL")
print("=" * 55)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print()
print("Model saved:")
print("model/disease_model.pkl")

print()
print("Symptoms saved:")
print("model/symptoms.pkl")

print("=" * 55)