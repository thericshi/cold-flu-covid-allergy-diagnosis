from symptom_finder import *
from prediction import predict
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def get_results():
    # Get the description of the person's symptoms from the user
    symptom_description = request.form['symptom_description']

    # Check for the symptoms in the description
    found_symptoms = check_symptoms(lemmatize(process_description(
        symptom_description)))

    if len(found_symptoms) == 0:
        found_symptoms = ["None"]

    result = predict(found_symptoms)

    return render_template('index.html', symptoms=found_symptoms, result=result[0])


if __name__ == '__main__':
    app.run()
