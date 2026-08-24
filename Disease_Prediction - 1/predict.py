import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("model/disease_model.pkl")
symptoms_list = joblib.load("model/symptoms.pkl")


print()
print("=" * 60)
print("           DISEASE PREDICTION SYSTEM")
print("=" * 60)


# ============================================================
# SHOW SYMPTOMS
# ============================================================

print("\nSelect your symptoms:\n")

for i, symptom in enumerate(symptoms_list, 1):
    print(f"{i}. {symptom.replace('_', ' ').title()}")


# ============================================================
# USER SELECTION
# ============================================================

user_input = input(
    "\nEnter symptom numbers separated by commas: "
)


try:

    selected_numbers = [
        int(x.strip())
        for x in user_input.split(",")
    ]

except ValueError:

    print("\nPlease enter valid numbers.")
    exit()


# ============================================================
# VALIDATE NUMBERS
# ============================================================

selected_symptoms = []

for number in selected_numbers:

    if 1 <= number <= len(symptoms_list):

        symptom = symptoms_list[number - 1]

        if symptom not in selected_symptoms:
            selected_symptoms.append(symptom)


if len(selected_symptoms) == 0:

    print("\nNo valid symptoms selected.")
    exit()


# ============================================================
# CREATE INPUT
# ============================================================

input_data = []

for symptom in symptoms_list:

    if symptom in selected_symptoms:
        input_data.append(1)
    else:
        input_data.append(0)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict([input_data])[0]

probabilities = model.predict_proba([input_data])[0]

confidence = max(probabilities) * 100


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("                    RESULT")
print("=" * 60)

print("\nSelected Symptoms:")

for symptom in selected_symptoms:
    print(
        "•",
        symptom.replace("_", " ").title()
    )

print(
    f"\nPredicted Disease: {prediction}"
)

print(
    f"Model Confidence: {confidence:.2f}%"
)

print("=" * 60)

print(
    "\n⚠️ Educational project only. "
    "This is not a medical diagnosis."
)