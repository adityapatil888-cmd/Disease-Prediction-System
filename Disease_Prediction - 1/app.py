from flask import Flask, render_template, request
import joblib


# ============================================================
# CREATE FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "model/disease_model.pkl"
)

symptoms_list = joblib.load(
    "model/symptoms.pkl"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    selected_symptoms = []


    # ========================================================
    # WHEN USER CLICKS PREDICT
    # ========================================================

    if request.method == "POST":

        selected_symptoms = request.form.getlist(
            "symptoms"
        )


        # ----------------------------------------------------
        # CHECK SELECTION
        # ----------------------------------------------------

        if len(selected_symptoms) == 0:

            prediction = "Please select at least one symptom."


        else:

            # ------------------------------------------------
            # CREATE INPUT VECTOR
            # ------------------------------------------------

            input_data = []

            for symptom in symptoms_list:

                if symptom in selected_symptoms:
                    input_data.append(1)
                else:
                    input_data.append(0)


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            prediction = model.predict(
                [input_data]
            )[0]


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            probabilities = model.predict_proba(
                [input_data]
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )


    # ========================================================
    # SEND DATA TO HTML
    # ========================================================

    return render_template(
        "index.html",
        symptoms=symptoms_list,
        prediction=prediction,
        confidence=confidence,
        selected_symptoms=selected_symptoms
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )