"""
Simple console application for prediction based on binary input
"""
import pickle
import pandas as pd

symptoms = ["COUGH", "MUSCLE_ACHES", "TIREDNESS", "SORE_THROAT", "RUNNY_NOSE", "STUFFY_NOSE", "FEVER", "NAUSEA",
            "VOMITING", "DIARRHEA", "SHORTNESS_OF_BREATH", "DIFFICULTY_BREATHING", "LOSS_OF_TASTE", "LOSS_OF_SMELL",
            "ITCHY_NOSE", "ITCHY_EYES", "ITCHY_MOUTH", "ITCHY_INNER_EAR", "SNEEZING", "PINK_EYE"]


def load_model():
    model = pickle.load(open("model.sav", 'rb'))
    return model


def get_symptoms():
    d = {}
    no_symptom = True

    for symptom in symptoms:
        valid = False
        while not valid:
            valid = True
            s_in = input("Do you experience: " + symptom + "?\n[y/n]: ")
            if s_in == "y":
                d[symptom] = [1]
                no_symptom = False
            elif s_in == "n":
                d[symptom] = [0]
            else:
                valid = False
                print("Input is not valid, please re-enter in the next line.")

    input_df = pd.DataFrame(d)
    return input_df, no_symptom


df, no_symptom = get_symptoms()
if no_symptom:
    print("Unable to diagnose since you have no symptoms.")
else:
    print("You most likely have: " + load_model().predict(df)[0])
