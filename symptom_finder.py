import re
import pandas as pd

import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')

correct_words = words.words()

# Create a list of symptoms to check for
symptoms = ["COUGH", "MUSCLE_ACHES", "TIREDNESS", "SORE_THROAT", "RUNNY_NOSE", "STUFFY_NOSE", "FEVER", "NAUSEA", "VOMITING", "DIARRHEA", "SHORTNESS_OF_BREATH", "DIFFICULTY_BREATHING", "LOSS_OF_TASTE", "LOSS_OF_SMELL", "ITCHY_NOSE", "ITCHY_EYES", "ITCHY_MOUTH", "ITCHY_INNER_EAR", "SNEEZING", "PINK_EYE"]

negations = ["not", "never", "no", "none"]


def check_negation_2gram(symptom, description):
    for negation in negations:
        if negation + " " + symptom in description:
            return True
        for string in similar_symptom_description(symptom):
            if negation + " " + string in description:
                return True
    return False


def check_negation(symptom, description):
    if check_negation_2gram(symptom, description):
        return True

    # Split the description into a list of individual words
    words = description.split(" ")

    # Loop through the words in the list
    for i, word in enumerate(words):
        # Check if the current word is a negation
        if word in negations:
            # Check if the symptom appears a few words later in the list
            if i+2 < len(words) and check_strings(similar_symptom_description(symptom), " ".join(words[i+1:i+3])):
                # If the symptom appears after the negation, return True
                return True
            if i+3 < len(words) and check_strings(similar_symptom_description(symptom), " ".join(words[i+1:i+4])):
                # If the symptom appears after the negation, return True
                return True
            if i+4 < len(words) and check_strings(similar_symptom_description(symptom), " ".join(words[i+1:i+5])):
                # If the symptom appears after the negation, return True
                return True

    # If the symptom is not negated, return False
    return False


def check_strings(l, s):
    """
    :return if any string in List l is a substring of String s
    """
    for string in l:
        if string in s:
            return True
    return False


def expand_contractions(sentence):
    # Regular expression to match contractions that end with "n't"
    nt_contraction_regex = r"\b[a-zA-Z]+n't\b"

    # Dictionary of contractions and their expansions
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "mightn't": "might not",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "weren't": "were not",
        "won't": "will not",
        "wouldn't": "would not"
    }

    # Use the regex to find all contractions in the sentence
    contractions_found = re.findall(nt_contraction_regex, sentence)

    # Replace each contraction with its expansion from the dictionary
    for contraction in contractions_found:
        sentence = sentence.replace(contraction, contractions[contraction])

    return sentence


def lemmatize(description):
    # Use the nltk word_tokenize() function to split the string into tokens
    tokens = nltk.word_tokenize(description)

    # Use the nltk pos_tag() function to identify the part of speech of each token
    tagged_tokens = nltk.pos_tag(tokens)

    processed = ""

    # Iterate over the tagged tokens
    for token in tagged_tokens:
        # Check if the token is a verb
        if 'VB' in token[1]:
            lemmatizer = nltk.WordNetLemmatizer()
            lemmatized = lemmatizer.lemmatize(token[0], pos='v')
            processed += lemmatized + " "
        else:
            processed += token[0] + " "
    return processed.strip()


def process_description(d):
    return expand_contractions(d.lower())


# Function to check for symptoms
def check_symptoms(symptom_description):
    # Create a list to store the symptoms found
    found_symptoms = []

    # Loop through the list of symptoms
    for symptom in symptoms:
        if not check_negation(symptom, symptom_description) and (check_strings(
                similar_symptom_description(symptom), symptom_description)):
            # If the symptom is present, add it to the list of found symptoms
            found_symptoms.append(symptom)

    # Return the list of found symptoms
    return found_symptoms


def convert_to_input(input_symptoms):
    print(input_symptoms)

    d = {}
    for sym in symptoms:
        if sym in input_symptoms:
            d[sym] = [1]
        else:
            d[sym] = [0]

    input_df = pd.DataFrame(d)
    return input_df


# Function to check for similar ways of describing the same symptom
def similar_symptom_description(symptom):
    # Create a dictionary to map symptoms to their similar descriptions
    symptom_map = {
        "COUGH": ["cough", "coughing", "coughed", "coughs", "persistent cough", "dry cough", "wet cough", "chesty cough"],
        "MUSCLE_ACHES": ["muscle aches", "aching muscles", "muscle pain", "muscle stiffness", "body ache", "body aches"],
        "TIREDNESS": ["tiredness", "tired", "fatigue", "exhaustion", "lethargy"],
        "SORE_THROAT": ["sore throat", "sore throats", "throat pain", "scratchy throat"],
        "RUNNY_NOSE": ["runny nose", "runny noses", "runny nostrils", "watery nose", "runny"],
        "STUFFY_NOSE": ["stuffy nose", "stuffy noses", "blocked nose", "congested nose", "stuffy"],
        "FEVER": ["fever", "fevers", "high temperature", "raised temperature"],
        "NAUSEA": ["nausea", "nauseous", "feeling sick", "stomach upset", "queasiness"],
        "VOMITING": ["vomiting", "vomited", "throwing up", "emptying stomach contents", "regurgitation"],
        "DIARRHEA": ["diarrhea", "diarrhoea", "loose stools", "watery stools", "frequent bowel movements"],
        "SHORTNESS_OF_BREATH": ["shortness of breath", "difficulty breathing", "breathing difficulty", "labored breathing", "dyspnea"],
        "DIFFICULTY_BREATHING": ["difficulty breathing", "breathing difficulty", "shortness of breath", "labored breathing", "dyspnea"],
        "LOSS_OF_TASTE": ["loss of taste", "taste loss", "taste buds not working", "altered taste", "distorted taste"],
        "LOSS_OF_SMELL": ["loss of smell", "smell loss", "smell not working", "altered smell", "distorted smell"],
        "ITCHY_NOSE": ["itchy nose", "itchy nostrils", "sneezing"],
        "ITCHY_EYES": ["itchy eyes", "itchy eyelids", "watery eyes", "red eyes"],
        "ITCHY_MOUTH": ["itchy mouth", "itchy tongue", "sore mouth", "tingling mouth"],
        "ITCHY_INNER_EAR": ["itchy inner ear", "itchy ears", "ear itchiness"],
        "SNEEZING": ["sneezing", "sneeze", "sneezing fits"],
        "PINK_EYE": ["pink eye", "pink eyes", "conjunctivitis", "red eye", "red eyes", "itchy eyes", "watery eyes", "discharge from eyes"]
    }

    # Return the similar description for the symptom
    return symptom_map[symptom]


def extract(description):
    return check_symptoms(lemmatize(process_description(description)))


if __name__ == '__main__':
    # Get the description of the person's symptoms from the user
    symptom_description = input("Please describe your symptoms: ")

    # Check for the symptoms in the description
    found_symptoms = extract(symptom_description)

    # Print out the found symptoms
    print("The following symptoms were found:")
    for symptom in found_symptoms:
        print(symptom)

    print(convert_to_input(found_symptoms))

