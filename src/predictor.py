import joblib

from src.preprocessing import clean_text


# =========================================
# LOAD MODEL FUNCTION
# =========================================

def load_model(model_name):

    if model_name == "lr":
        return joblib.load("models/logistic_regression.pkl")

    elif model_name == "nb":
        return joblib.load("models/naive_bayes.pkl")

    elif model_name == "svm":
        return joblib.load("models/svm_model.pkl")

    else:
        raise ValueError("Invalid Model Name")


# =========================================
# LOAD VECTORIZER
# =========================================

vectorizer = joblib.load("models/vectorizer.pkl")


# =========================================
# PREDICTION FUNCTION
# =========================================

def predict_review(text, model_name="svm"):

    model = load_model(model_name)

    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    if prediction[0] == 0:
        return "Fake Review"

    return "Genuine Review"