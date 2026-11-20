
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

print("=" * 60)
print("FAKE NEWS DETECTION - TF-IDF FEATURE EXTRACTION")
print("=" * 60)

# STEP 1: Load cleaned dataset

print("\nSTEP 1: Loading dataset...")

Data = pd.read_csv("dataset/combined_news.csv")

X = Data["content"].fillna("")
Y = Data["label"]

print("Total articles:", len(Data))
print("Total fake articles:", (Y == 0).sum())
print("Total real articles:", (Y == 1).sum())

# STEP 2: Split dataset

print("\nSTEP 2: Splitting dataset...")

XTrain, XTest, YTrain, YTest = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("Training articles:", len(XTrain))
print("Testing articles:", len(XTest))

# STEP 3: Initialize TF-IDF

print("\nSTEP 3: Initializing TF-IDF...")

Vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

# STEP 4: Learn vocabulary from training data

print("\nSTEP 4: Learning vocabulary...")

XTrainTFIDF = Vectorizer.fit_transform(XTrain)

# Transform test data using training vocabulary only

XTestTFIDF = Vectorizer.transform(XTest)

# STEP 5: Display feature information

print("\nSTEP 5: Feature extraction results...")

print("Training matrix:", XTrainTFIDF.shape)
print("Testing matrix:", XTestTFIDF.shape)

print("Unique vocabulary size:",
      len(Vectorizer.vocabulary_))

print("Training matrix non-zero values:",
      XTrainTFIDF.nnz)

# STEP 6: Save vectorizer and matrices

print("\nSTEP 6: Saving extracted features...")

joblib.dump(
    Vectorizer,
    "models/tfidf_vectorizer.pkl"
)

save_npz(
    "models/X_train.npz",
    XTrainTFIDF
)

save_npz(
    "models/X_test.npz",
    XTestTFIDF
)

np.save(
    "models/y_train.npy",
    YTrain.to_numpy()
)

np.save(
    "models/y_test.npy",
    YTest.to_numpy()
)

print("\nAll features saved successfully!")

print("\nSaved files:")
print("tfidf_vectorizer.pkl")
print("X_train.npz")
print("X_test.npz")
print("y_train.npy")
print("y_test.npy")

print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION COMPLETED")
print("=" * 60)
