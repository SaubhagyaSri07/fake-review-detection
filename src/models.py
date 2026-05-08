from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def train_logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")
    print("LOGISTIC REGRESSION")
    print("==============================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    return model


def train_naive_bayes(X_train, X_test, y_train, y_test):

    model = MultinomialNB()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")
    print("NAIVE BAYES")
    print("==============================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    return model


def train_svm(X_train, X_test, y_train, y_test):

    model = LinearSVC()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n==============================")
    print("SVM")
    print("==============================")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    return model