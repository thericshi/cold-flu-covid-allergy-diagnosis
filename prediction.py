import pickle
import pandas as pd
from symptom_finder import extract, convert_to_input


def second_max(prob_list):
    """

    :param prob_list:
    :return: second-largest item in the list
    """
    largest = -float('inf')
    second_largest = -float('inf')

    for number in prob_list:
        if number > largest:
            second_largest = largest
            largest = number
        elif number > second_largest:
            second_largest = number

    return second_largest


def load_model():
    model = pickle.load(open("lrmodel.sav", 'rb'))
    return model


def predict(input_symptoms):
    classes = list(load_model().classes_)
    prediction = load_model().predict(convert_to_input(input_symptoms))[0]
    prob = load_model().predict_proba(convert_to_input(input_symptoms))[0]
    confidence = round(prob[classes.index(prediction)], 2) * 100

    second_pred = classes[list(prob).index(second_max(list(prob)))]
    second_conf = round(second_max(list(prob)), 2) * 100

    return prediction, confidence, second_pred, second_conf


if __name__ in "__main__":
    symptom_description = input("Please describe your symptoms: ")

    found_symptoms = extract(symptom_description)
    print(predict(found_symptoms))
