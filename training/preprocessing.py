
import pandas as pd

print("=" * 60)
print("FAKE NEWS DETECTION - DATA PREPROCESSING")
print("=" * 60)

# STEP 1: Load datasets
print("\nSTEP 1: Loading datasets...")

FakeData = pd.read_csv("dataset/Fake.csv")
RealData = pd.read_csv("dataset/True.csv")

print("Fake news records:", len(FakeData))
print("Real news records:", len(RealData))

# STEP 2: Assign labels
print("\nSTEP 2: Assigning labels...")

FakeData["label"] = 0
RealData["label"] = 1

print("0 = Fake News")
print("1 = Real News")

# STEP 3: Combine datasets
print("\nSTEP 3: Combining datasets...")

Data = pd.concat(
    [FakeData, RealData],
    ignore_index=True
)

print("Original combined records:", len(Data))

# STEP 4: Analyze data quality
print("\nSTEP 4: Analyzing data quality...")

print("Exact duplicate rows:",
      Data.duplicated().sum())

print("Duplicate title and text pairs:",
      Data.duplicated(subset=["title", "text"]).sum())

TitleBlank = (
    Data["title"].fillna("").astype(str).str.strip() == ""
)

TextBlank = (
    Data["text"].fillna("").astype(str).str.strip() == ""
)

print("Empty titles:", TitleBlank.sum())
print("Empty article texts:", TextBlank.sum())

# STEP 5: Check conflicting labels
LabelGroups = Data.groupby(
    ["title", "text"],
    dropna=False
)["label"].nunique()

ConflictingGroups = LabelGroups[LabelGroups > 1]

print("Conflicting label groups:",
      len(ConflictingGroups))

# STEP 6: Remove exact duplicates
print("\nSTEP 6: Removing duplicate records...")

Before = len(Data)

Data = Data.drop_duplicates().copy()

ExactRemoved = Before - len(Data)

print("Exact duplicates removed:", ExactRemoved)

# STEP 7: Remove missing and empty articles
print("\nSTEP 7: Removing empty articles...")

Before = len(Data)

Data = Data.dropna(
    subset=["title", "text"]
).copy()

Data["title"] = Data["title"].astype(str).str.strip()
Data["text"] = Data["text"].astype(str).str.strip()

Data = Data[
    (Data["title"] != "") &
    (Data["text"] != "")
].copy()

EmptyRemoved = Before - len(Data)

print("Empty articles removed:", EmptyRemoved)

# STEP 8: Remove repeated articles
print("\nSTEP 8: Removing repeated articles...")

Before = len(Data)

Data = Data.drop_duplicates(
    subset=["title", "text"]
).copy()

RepeatedRemoved = Before - len(Data)

print("Repeated articles removed:", RepeatedRemoved)

# STEP 9: Create combined content
print("\nSTEP 9: Creating text features...")

Data["content"] = (
    Data["title"] + " " + Data["text"]
)

# STEP 10: Shuffle dataset
print("\nSTEP 10: Shuffling dataset...")

Data = Data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# STEP 11: Save processed dataset
print("\nSTEP 11: Saving processed dataset...")

Data.to_csv(
    "dataset/combined_news.csv",
    index=False
)

# STEP 12: Final report
print("\n" + "=" * 60)
print("FINAL DATASET REPORT")
print("=" * 60)

print("Total articles:", len(Data))

print("\nLabel distribution:")
print(Data["label"].value_counts())

print("\nClass percentages:")
print(
    (Data["label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nRemaining exact duplicates:",
      Data.duplicated().sum())

print("Remaining duplicate articles:",
      Data.duplicated(
          subset=["title", "text"]
      ).sum())

print("\nFinal dataset shape:", Data.shape)

print("\nFinal columns:")
print(Data.columns.tolist())

print("\nSample articles:")
print(
    Data[["title", "label"]].head(5).to_string(index=False)
)

print("\nProcessed dataset saved at:")
print("dataset/combined_news.csv")

print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
