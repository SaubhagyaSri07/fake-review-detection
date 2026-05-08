from src.predictor import predict_review


print("\nFake Review Detection System\n")

print("Choose Model:")
print("1. Logistic Regression")
print("2. Naive Bayes")
print("3. SVM\n")

choice = input("Enter Choice: ")


if choice == "1":
    model_name = "lr"

elif choice == "2":
    model_name = "nb"

else:
    model_name = "svm"


while True:

    review = input("\nEnter Review (type exit/quit to quit): ")

    if review.lower() in ["exit", "quit"]:
        print("\nProgram Closed")
        break

    result = predict_review(review, model_name)

    print("Prediction:", result)