
import pandas as pd

print("=" * 65)
print("TRUTHLENS - DATASET BIAS ANALYSIS")
print("=" * 65)

Data = pd.read_csv("dataset/combined_news.csv")

print("\nSUBJECT DISTRIBUTION")
print("-" * 65)

SubjectTable = pd.crosstab(
    Data["subject"],
    Data["label"],
    margins=True
)

SubjectTable.columns = ["Fake News", "Real News"] if len(SubjectTable.columns) == 2 else SubjectTable.columns

print(SubjectTable)

print("\nSUBJECT-WISE FAKE/REAL PERCENTAGES")
print("-" * 65)

Distribution = pd.crosstab(
    Data["subject"],
    Data["label"],
    normalize="index"
) * 100

Distribution = Distribution.rename(
    columns={0: "Fake %", 1: "Real %"}
)

print(Distribution.round(2))

print("\nOVERALL LABEL DISTRIBUTION")
print("-" * 65)

print(Data["label"].value_counts())
print(Data["label"].value_counts(normalize=True).mul(100).round(2))

print("\n" + "=" * 65)
print("DATASET ANALYSIS COMPLETED")
print("=" * 65)
