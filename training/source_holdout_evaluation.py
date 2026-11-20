
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 65)
print("FAKE NEWS DETECTION - SOURCE HOLDOUT EVALUATION")
print("=" * 65)

# STEP 1: Load dataset

Data = pd.read_csv("dataset/combined_news.csv")

print("\nTotal articles:", len(Data))

# STEP 2: Define held-out subjects

# These subjects will be completely excluded from training

HeldOutSubjects = ["left-news", "worldnews"]

TrainData = Data[
    ~Data["subject"].isin(HeldOutSubjects)
].copy()

TestData = Data[
    Data["subject"].isin(HeldOutSubjects)
].copy()

print("\nSTEP 2: SUBJECT SPLIT")

print("Training subjects:")
print(sorted(TrainData["subject"].unique()))

print("\nTesting subjects:")
print(sorted(TestData["subject"].unique()))

print("\nTraining records:", len(TrainData))
print("Testing records:", len(TestData))

# STEP 3: Check class distribution

print("\nSTEP 3: LABEL DISTRIBUTION")

print("Training:")
print(TrainData["label"].value_counts())

print("\nTesting:")
print(TestData["label"].value_counts())

# Check that both classes exist

if TrainData["label"].nunique() < 2:
    raise ValueError("Training data must contain both classes.")

if TestData["label"].nunique() < 2:
    raise ValueError("Testing data must contain both classes.")

# STEP 4: Prepare text

XTrain = TrainData["content"].fillna("")
YTrain = TrainData["label"]

XTest = TestData["content"].fillna("")
YTest = TestData["label"]

# STEP 5: Create fresh TF-IDF vectorizer

print("\nSTEP 5: TF-IDF FEATURE EXTRACTION")

Vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

XTrainTFIDF = Vectorizer.fit_transform(XTrain)

XTestTFIDF = Vectorizer.transform(XTest)

print("Training matrix:", XTrainTFIDF.shape)
print("Testing matrix:", XTestTFIDF.shape)

# STEP 6: Train fresh Linear SVM

print("\nSTEP 6: TRAINING LINEAR SVM")

Model = LinearSVC(
    C=1.0,
    random_state=42
)

Model.fit(XTrainTFIDF, YTrain)

print("Model training completed!")

# STEP 7: Make predictions

print("\nSTEP 7: MODEL EVALUATION")

Predictions = Model.predict(XTestTFIDF)

Accuracy = accuracy_score(YTest, Predictions)
Precision = precision_score(YTest, Predictions, zero_division=0)
Recall = recall_score(YTest, Predictions, zero_division=0)
F1 = f1_score(YTest, Predictions, zero_division=0)

print("Accuracy:", round(Accuracy * 100, 2), "%")
print("Precision:", round(Precision * 100, 2), "%")
print("Recall:", round(Recall * 100, 2), "%")
print("F1 Score:", round(F1 * 100, 2), "%")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        YTest,
        Predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    )
)

print("CONFUSION MATRIX")

print(
    confusion_matrix(
        YTest,
        Predictions,
        labels=[0, 1]
    )
)

# STEP 8: Save experimental model

import joblib

joblib.dump(
    Model,
    "models/source_holdout_svm.pkl"
)

joblib.dump(
    Vectorizer,
    "models/source_holdout_vectorizer.pkl"
)

print("\nExperimental model saved successfully!")

print("\n" + "=" * 65)
print("SOURCE HOLDOUT EVALUATION COMPLETED")
print("=" * 65)
