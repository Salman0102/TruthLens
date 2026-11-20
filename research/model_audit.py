
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL RELIABILITY AUDIT")
print("=" * 60)

# STEP 1: Load cleaned dataset

Data = pd.read_csv("dataset/combined_news.csv")

X = Data["content"].fillna("")
Y = Data["label"]

# STEP 2: Recreate original train-test split

XTrain, XTest, YTrain, YTest = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("\nSTEP 1: DATA SPLIT")
print("Training records:", len(XTrain))
print("Testing records:", len(XTest))

# STEP 3: Check normalized text overlap

print("\nSTEP 2: CHECKING TEXT OVERLAP")

TrainText = set(
    XTrain.str.lower().str.split().str.join(" ")
)

TestText = set(
    XTest.str.lower().str.split().str.join(" ")
)

Overlap = TrainText.intersection(TestText)

print("Overlapping normalized articles:", len(Overlap))

# STEP 4: Check subject distribution

print("\nSTEP 3: SUBJECT DISTRIBUTION")

SubjectTable = pd.crosstab(
    Data["subject"],
    Data["label"],
    normalize="index"
) * 100

SubjectTable.columns = ["Fake %", "Real %"]

print(SubjectTable.round(2).to_string())

# STEP 5: Check label distribution

print("\nSTEP 4: LABEL DISTRIBUTION")

print(
    Data["label"].value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# STEP 6: Check text lengths

print("\nSTEP 5: ARTICLE LENGTH ANALYSIS")

Data["text_length"] = Data["content"].str.len()

print(
    Data.groupby("label")["text_length"]
    .describe()
    .round(2)
)

# STEP 7: Final audit summary

print("\n" + "=" * 60)
print("MODEL RELIABILITY AUDIT COMPLETED")
print("=" * 60)

print("Normalized overlap:", len(Overlap))
print("Dataset size:", len(Data))

print("\nNote:")
print("This audit checks exact normalized text overlap")
print("and potential subject-related dataset bias.")
print("It does not prove real-world generalization.")
