import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from scipy.sparse import hstack

from src.vectorizer import create_tfidf


def train_metadata_model(df):

    # Encode category
    encoder = LabelEncoder()

    df['category_encoded'] = encoder.fit_transform(df['category'])

    X_meta = df[['rating', 'category_encoded']]

    y = df['label']

    X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        df['clean_text'],
        X_meta,
        y,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF
    vectorizer, X_train_tfidf, X_test_tfidf = create_tfidf(
        X_train_text,
        X_test_text
    )

    # Combine
    X_train_combined = hstack([
        X_train_tfidf,
        X_train_meta
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        X_test_meta
    ])

    # Train model
    model = LinearSVC()

    model.fit(X_train_combined, y_train)

    predictions = model.predict(X_test_combined)

    print("\n==============================")
    print("SVM + METADATA")
    print("==============================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    return model