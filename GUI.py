from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model and feature extraction objects
with open('spam_classifier.pkl', 'rb') as file:
    model, feature_extraction = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    input_the_mail = [message]
    input_data_features = feature_extraction.transform(input_the_mail)
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        result = "HAM MAIL"
    else:
        result = "SPAM MAIL"

    return render_template('index.html', message=message, result=result)


if __name__ == '__main__':
    app.run(debug=True)
