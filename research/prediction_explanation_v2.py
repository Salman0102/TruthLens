
import joblib
import numpy as np

print("=" * 65)
print("TRUTHLENS - COMPLETE PREDICTION EXPLANATION")
print("=" * 65)

Model = joblib.load("models/best_model.pkl")
Vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

Articles = {
    "Scientific reporting": """Researchers from several universities have published a study examining changes in global temperatures over recent decades. The research uses historical climate observations and computer simulations to evaluate long-term trends.""",

    "Flooding report": """Heavy rainfall caused severe flooding in several districts, damaging homes and disrupting transportation services. Emergency response teams were deployed to assist affected residents.""",

    "Sensational headline": """SHOCKING! Secret Government Cure Finally EXPOSED! You won't believe what happens next! Share this information before it gets deleted."""
}

FeatureNames = Vectorizer.get_feature_names_out()
Coefficients = Model.coef_[0]
Intercept = float(Model.intercept_[0])

for Name, Article in Articles.items():

    Matrix = Vectorizer.transform([Article])
    Contributions = Matrix.toarray()[0] * Coefficients

    ActiveIndices = np.flatnonzero(Matrix.toarray()[0])

    PositiveTotal = Contributions[Contributions > 0].sum()
    NegativeTotal = Contributions[Contributions < 0].sum()

    CalculatedScore = Intercept + Contributions.sum()
    ActualScore = float(Model.decision_function(Matrix)[0])

    Prediction = Model.predict(Matrix)[0]

    print("\n" + "=" * 65)
    print("ARTICLE:", Name)
    print("PREDICTION:", "REAL NEWS" if Prediction == 1 else "FAKE NEWS")

    print("\nDECISION BREAKDOWN")
    print("-" * 65)
    print("Model intercept:", round(Intercept, 4))
    print("Positive contributions:", round(float(PositiveTotal), 4))
    print("Negative contributions:", round(float(NegativeTotal), 4))
    print("Calculated score:", round(float(CalculatedScore), 4))
    print("Actual score:", round(ActualScore, 4))
    print("Reconstruction difference:", round(abs(CalculatedScore - ActualScore), 8))

    SortedIndices = sorted(
        ActiveIndices,
        key=lambda Index: abs(Contributions[Index]),
        reverse=True
    )

    print("\nTOP 10 POSITIVE FEATURES")
    for Index in sorted(
        ActiveIndices,
        key=lambda Index: Contributions[Index],
        reverse=True
    )[:10]:
        print(f"{FeatureNames[Index]:30s} {Contributions[Index]:+.4f}")

    print("\nTOP 10 NEGATIVE FEATURES")
    for Index in sorted(
        ActiveIndices,
        key=lambda Index: Contributions[Index]
    )[:10]:
        print(f"{FeatureNames[Index]:30s} {Contributions[Index]:+.4f}")

print("\n" + "=" * 65)
print("COMPLETE EXPLANATION FINISHED")
print("=" * 65)
