
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 65)
print("TRUTHLENS - IMPROVED MODEL EXPERIMENT")
print("=" * 65)

# 1. Load dataset
Data = pd.read_csv("dataset/combined_news.csv")

Data["content"] = Data["content"].fillna("").astype(str)
Data["subject"] = Data["subject"].fillna("")

# 2. Keep the same held-out subjects
TestSubjects = ["left-news", "worldnews"]

TrainData = Data[~Data["subject"].isin(TestSubjects)].copy()
TestData = Data[Data["subject"].isin(TestSubjects)].copy()

print("\nDATA SPLIT")
print("-" * 65)
print("Training records:", len(TrainData))
print("Testing records:", len(TestData))
print("Training subjects:", sorted(TrainData["subject"].unique()))
print("Testing subjects:", TestSubjects)

# 3. Split training data into training and validation
TrainPart, ValidationPart = train_test_split(
    TrainData,
    test_size=0.20,
    random_state=42,
    stratify=TrainData["label"]
)

YTrain = TrainPart["label"].values
YValidation = ValidationPart["label"].values
YTest = TestData["label"].values

# 4. Text cleaning for candidate model
def CleanText(Text):
    Text = Text.lower()

    Patterns = [
        r"https?://\S+|www\.\S+",
        r"\bread more\b",
        r"\bfeatured image\b",
        r"\bimage via\b",
        r"\bgetty images?\b",
        r"\breuters\b",
        r"\bphoto credit\b",
        r"\ball rights reserved\b"
    ]

    for Pattern in Patterns:
        Text = re.sub(Pattern, " ", Text)

    Text = re.sub(r"\s+", " ", Text).strip()

    return Text

# 5. Evaluation function
def EvaluateModel(Name, Model, XTest, YActual):
    Predictions = Model.predict(XTest)

    Accuracy = accuracy_score(YActual, Predictions)
    BalancedAccuracy = balanced_accuracy_score(YActual, Predictions)
    MacroF1 = f1_score(YActual, Predictions, average="macro")
    FakeF1 = f1_score(YActual, Predictions, pos_label=0)
    RealF1 = f1_score(YActual, Predictions, pos_label=1)

    print("\n" + "=" * 65)
    print(Name)
    print("=" * 65)

    print(f"Accuracy:          {Accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {BalancedAccuracy * 100:.2f}%")
    print(f"Macro F1:          {MacroF1 * 100:.2f}%")
    print(f"Fake News F1:      {FakeF1 * 100:.2f}%")
    print(f"Real News F1:      {RealF1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(YActual, Predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        YActual,
        Predictions,
        labels=[0, 1],
        target_names=["Fake News", "Real News"],
        zero_division=0
    ))

    return {
        "accuracy": Accuracy,
        "balanced_accuracy": BalancedAccuracy,
        "macro_f1": MacroF1,
        "fake_f1": FakeF1,
        "real_f1": RealF1
    }

# 6. Baseline model: original text
print("\nTraining baseline model...")

BaselineVectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

XBaselineTrain = BaselineVectorizer.fit_transform(
    TrainPart["content"]
)

XBaselineTest = BaselineVectorizer.transform(
    TestData["content"]
)

BaselineModel = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42
)

BaselineModel.fit(XBaselineTrain, YTrain)

BaselineResults = EvaluateModel(
    "BASELINE - ORIGINAL TEXT",
    BaselineModel,
    XBaselineTest,
    YTest
)

# 7. Candidate model: cleaned text
print("\nPreparing cleaned text...")

TrainTexts = TrainPart["content"].apply(CleanText)
ValidationTexts = ValidationPart["content"].apply(CleanText)
TestTexts = TestData["content"].apply(CleanText)

CandidateVectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

XCandidateTrain = CandidateVectorizer.fit_transform(TrainTexts)
XCandidateValidation = CandidateVectorizer.transform(ValidationTexts)

# 8. Select hyperparameters using validation only
print("\nHYPERPARAMETER EXPERIMENT")
print("-" * 65)

BestScore = -1
BestC = None
BestWeight = None

for CValue in [0.5, 1.0, 2.0]:
    for Weight in [None, "balanced"]:

        Model = LinearSVC(
            C=CValue,
            class_weight=Weight,
            random_state=42
        )

        Model.fit(XCandidateTrain, YTrain)

        ValidationPredictions = Model.predict(XCandidateValidation)

        Score = f1_score(
            YValidation,
            ValidationPredictions,
            average="macro"
        )

        print(
            f"C={CValue}, class_weight={Weight}, "
            f"Validation Macro F1={Score * 100:.2f}%"
        )

        if Score > BestScore:
            BestScore = Score
            BestC = CValue
            BestWeight = Weight

print("\nSelected C:", BestC)
print("Selected class weight:", BestWeight)

# 9. Retrain candidate on all non-held-out subjects
print("\nTraining final candidate model...")

FinalTexts = TrainData["content"].apply(CleanText)

FinalVectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
    dtype=np.float32
)

XFinalTrain = FinalVectorizer.fit_transform(FinalTexts)
XFinalTest = FinalVectorizer.transform(TestTexts)

CandidateModel = LinearSVC(
    C=BestC,
    class_weight=BestWeight,
    random_state=42
)

CandidateModel.fit(XFinalTrain, TrainData["label"].values)

CandidateResults = EvaluateModel(
    "CANDIDATE - CLEANED TEXT",
    CandidateModel,
    XFinalTest,
    YTest
)

# 10. Compare results
print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(f"{'Metric':<22}{'Baseline':>15}{'Candidate':>15}")

for Metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "fake_f1",
    "real_f1"
]:
    print(
        f"{Metric:<22}"
        f"{BaselineResults[Metric] * 100:>14.2f}%"
        f"{CandidateResults[Metric] * 100:>14.2f}%"
    )

# 11. Save candidate separately - do not replace production model
joblib.dump(CandidateModel, "models/candidate_model.pkl")
joblib.dump(FinalVectorizer, "models/candidate_vectorizer.pkl")

print("\nCandidate model saved successfully.")
print("File: models/candidate_model.pkl")
print("File: models/candidate_vectorizer.pkl")

print("\nExisting production model has NOT been changed.")
print("=" * 65)
print("EXPERIMENT COMPLETED")
print("=" * 65)
