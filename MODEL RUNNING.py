import pickle

# Load the saved model from the file
with open('spam_classifier.pkl', 'rb') as file:
    model, feature_extraction = pickle.load(file)

# Now you can use the loaded model for prediction
input_the_mail = ["how are you?"]
input_data_features = feature_extraction.transform(input_the_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("HAM MAIL")
else:
    print("SPAM MAIL")
