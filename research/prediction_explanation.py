
import joblib
import numpy as np

print("=" * 65)
print("TRUTHLENS - PREDICTION EXPLANATION")
print("=" * 65)

Model = joblib.load("models/best_model.pkl")
Vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

Articles = {
    "Scientific reporting":
    """Researchers from several universities have published a
    study examining changes in global temperatures over recent
    decades. The research uses historical climate observations
    and computer simulations to evaluate long-term trends.""",

    "Flooding report":
    """Heavy rainfall caused severe flooding in several districts,
    damaging homes and disrupting transportation services.
    Emergency response teams were deployed to assist affected
    residents.""",

    "Sensational headline":
    """SHOCKING! Secret Government Cure Finally EXPOSED!
    You won't believe what happens next! Share this information
    before it gets deleted."""
}

FeatureNames = Vectorizer.get_feature_names_out()
Coefficients = Model.coef_[0]

for Name, Article in Articles.items():

    Matrix = Vectorizer.transform([Article])

    Contributions = Matrix.toarray()[0] * Coefficients

    ActiveIndices = np.flatnonzero(Matrix.toarray()[0])

    SortedIndices = sorted(
        ActiveIndices,
        key=lambda Index: abs(Contributions[Index]),
        reverse=True
    )

    Prediction = Model.predict(Matrix)[0]
    Score = Model.decision_function(Matrix)[0]

    print("\n" + "=" * 65)
    print("ARTICLE:", Name)
    print("PREDICTION:", "REAL NEWS" if Prediction == 1 else "FAKE NEWS")
    print("DECISION SCORE:", round(float(Score), 4))

    print("\nTOP FEATURES PUSHING THIS PREDICTION")
    print("-" * 65)

    for Index in SortedIndices[:15]:
        print(
            f"{FeatureNames[Index]:35s}"
            f"{Contributions[Index]:+.4f}"
        )

print("\n" + "=" * 65)
print("EXPLANATION COMPLETED")
print("=" * 65)
