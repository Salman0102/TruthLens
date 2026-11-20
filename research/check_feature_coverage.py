import pandas as pd
import joblib
import numpy as np

df = pd.read_csv("dataset/combined_news.csv")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

texts = df["content"].fillna("").astype(str).sample(
    n=min(1000, len(df)), random_state=42
)

counts = vectorizer.transform(texts).getnnz(axis=1)

print("Articles tested:", len(counts))
print("Zero-feature articles:", np.sum(counts == 0))
print("Minimum:", np.min(counts))
print("25th percentile:", np.percentile(counts, 25))
print("Median:", np.median(counts))
print("75th percentile:", np.percentile(counts, 75))
print("Maximum:", np.max(counts))