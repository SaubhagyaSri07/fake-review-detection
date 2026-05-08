import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text
from src.vectorizer import create_tfidf

from src.models import (
    train_logistic_regression,
    train_naive_bayes,
    train_svm
)

from src.metadata_model import train_metadata_model


# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("data/fake reviews dataset.csv")

print("\nDataset Loaded Successfully")

print("Dataset Shape:", df.shape)


# =========================================
# CLEAN TEXT
# =========================================

df['clean_text'] = df['text_'].apply(clean_text)

print("\nText Cleaning Completed")


# =========================================
# ENCODE LABELS
# =========================================

df['label'] = df['label'].map({
    'CG': 0,
    'OR': 1
})

print("\nLabels Encoded")


# =========================================
# TRAIN TEST SPLIT
# =========================================

X = df['clean_text']

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed")


# =========================================
# TF-IDF
# =========================================

vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf(
    X_train,
    X_test
)

print("\nTF-IDF Vectorization Completed")


# =========================================
# TRAIN MODELS
# =========================================

lr_model = train_logistic_regression(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
)

nb_model = train_naive_bayes(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
)

svm_model = train_svm(
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
)

# =========================================
# SAVE MODEL
# =========================================

os.makedirs("models", exist_ok=True)

joblib.dump(lr_model, "models/logistic_regression.pkl")

joblib.dump(nb_model, "models/naive_bayes.pkl")

joblib.dump(svm_model, "models/svm_model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nAll Models Saved Successfully")

# =========================================
# METADATA MODEL
# =========================================

train_metadata_model(df)


print("\nProject Execution Completed")