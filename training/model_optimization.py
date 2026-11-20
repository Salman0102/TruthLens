
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL OPTIMIZATION")
print("=" * 60)

os.makedirs("models", exist_ok=True)

Data = pd.read_csv("dataset/combined_news.csv")

HeldOutSubjects = ["left-news", "worldnews"]

TrainData = Data[
    ~Data["subject"].isin(HeldOutSubjects)
].copy()

TestData = Data[
    Data["subject"].isin(HeldOutSubjects)
].copy()

XTrain, XValidation, YTrain, YValidation = train_test_split(
    TrainData["content"].fillna(""),
    TrainData["label"],
    test_size=0.20,
    random_state=42,
    stratify=TrainData["label"]
)

XTest = TestData["content"].fillna("")
YTest = TestData["label"]

print("\nDataset Information")
print("Training records:", len(XTrain))
print("Validation records:", len(XValidation))
print("Testing records:", len(XTest))

Vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

print("\nExtracting TF-IDF Features...")

XTrainTFIDF = Vectorizer.fit_transform(XTrain)
XTestTFIDF = Vectorizer.transform(XTest)

print("Training matrix:", XTrainTFIDF.shape)
print("Testing matrix:", XTestTFIDF.shape)

Models = {
    "Original Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    ),

    "Balanced Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    )
}

Results = []

for Name, Model in Models.items():

    print("\n" + "=" * 60)
    print("MODEL:", Name)
    print("=" * 60)

    Model.fit(XTrainTFIDF, YTrain)

    Predictions = Model.predict(XTestTFIDF)

    Accuracy = accuracy_score(YTest, Predictions)
    BalancedAccuracy = balanced_accuracy_score(YTest, Predictions)
    MacroF1 = f1_score(YTest, Predictions, average="macro")
    FakeF1 = f1_score(YTest, Predictions, pos_label=0)

    print("Accuracy:", round(Accuracy * 100, 2), "%")
    print("Balanced Accuracy:", round(BalancedAccuracy * 100, 2), "%")
    print("Macro F1:", round(MacroF1 * 100, 2), "%")
    print("Fake News F1:", round(FakeF1 * 100, 2), "%")

    print("\nClassification Report:")

    print(classification_report(
        YTest,
        Predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))

    print("Confusion Matrix:")

    print(confusion_matrix(
        YTest,
        Predictions,
        labels=[0, 1]
    ))

    Results.append({
        "Model": Name,
        "Accuracy": Accuracy * 100,
        "Balanced Accuracy": BalancedAccuracy * 100,
        "Macro F1": MacroF1 * 100,
        "Fake F1": FakeF1 * 100
    })

    if Name == "Balanced Linear SVM":

        joblib.dump(
            Model,
            "models/best_model.pkl"
        )

        joblib.dump(
            Vectorizer,
            "models/tfidf_vectorizer.pkl"
        )

        print("\nOptimized model saved successfully!")
        print("Vectorizer saved successfully!")

print("\n" + "=" * 60)
print("MODEL OPTIMIZATION COMPARISON")
print("=" * 60)

ResultsData = pd.DataFrame(Results)

print(ResultsData.to_string(index=False))

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)

print("Selected Model: Balanced Linear SVM")
print("Model Path: models/best_model.pkl")
print("Vectorizer Path: models/tfidf_vectorizer.pkl")

print("\nOptimization experiment completed!")
