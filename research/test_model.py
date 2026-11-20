import joblib

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

tests = [
    "Breaking news: Government announces new education policy.",
    "The weather is sunny today.",
    "12345 !!!",
    "Scientists publish a new research paper on climate change."
]

for text in tests:
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    score = model.decision_function(features)[0]

    label = "Real" if prediction == 1 else "Fake"

    print("\nArticle:", text)
    print("Prediction:", label)
    print("Decision score:", round(float(score), 4))
