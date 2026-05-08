from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf(X_train, X_test):

    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf