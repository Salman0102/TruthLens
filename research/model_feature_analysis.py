
import joblib
import numpy as np

print("=" * 60)
print("TRUTHLENS - MODEL FEATURE ANALYSIS")
print("=" * 60)

Model = joblib.load("models/best_model.pkl")
Vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

FeatureNames = Vectorizer.get_feature_names_out()
Coefficients = Model.coef_[0]

TopFakeIndices = np.argsort(Coefficients)[:25]
TopRealIndices = np.argsort(Coefficients)[-25:][::-1]

print("\nTOP 25 FEATURES ASSOCIATED WITH FAKE NEWS")
print("-" * 60)

for Index in TopFakeIndices:
    print(f"{FeatureNames[Index]:35s} {Coefficients[Index]:.4f}")

print("\nTOP 25 FEATURES ASSOCIATED WITH REAL NEWS")
print("-" * 60)

for Index in TopRealIndices:
    print(f"{FeatureNames[Index]:35s} {Coefficients[Index]:.4f}")

print("\n" + "=" * 60)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 60)
