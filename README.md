# Cold Flu COVID-19 and Allergy Diagnosis

This Flask application predicts a patient's most likely respiratory condition (common cold, flu, COVID-19, or allergy) with a 96% accuracy based a description of their symptoms. Various natural language processing techniques are used to ensure an accurate extraction of symptoms from the description. 

## Demo
A short video (11 sec) demonstration of the basic functionalities of the application

https://user-images.githubusercontent.com/99521397/208178609-fb63ed6c-c6e5-4ac2-b9e0-8244beca6422.mov

## Machine learning models
The following models are compared using Scikit-learn, with their 5-fold cross-validation scores: Support vector machine(96%), K-nearest neighbour(85%), Multi-layer perceptron classifier(95%), Logistic regression(96%), decision tree(92%), random forest(92%), naive bayes(96%).

The model used in the Flask application is the Logistic regression model. 

## Requirements
The following list of packages is required to run the flask application:
- Python 3
- nltk
- pandas
- regex
- scikit-learn
- flask

## Usage
To run the flask application:
```python flask_symptom.py```
