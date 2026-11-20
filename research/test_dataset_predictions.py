import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("dataset/combined_news.csv")
model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

sample = df.sample(n=1000, random_state=42)

X = vectorizer.transform(sample["content"].fillna("").astype(str))
y_true = sample["label"].values
y_pred = model.predict(X)

print("Articles tested:", len(sample))
print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    zero_division=0
))

print("\nPrediction distribution:")
print(pd.Series(y_pred).value_counts().rename(index={0: "Fake", 1: "Real"}))