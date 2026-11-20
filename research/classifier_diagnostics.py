
import joblib
import pandas as pd
import numpy as np

print("=" * 60)
print("TRUTHLENS - CLASSIFIER DIAGNOSTICS")
print("=" * 60)

Model = joblib.load("models/best_model.pkl")
Vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

TestArticles = {
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

for Name, Article in TestArticles.items():

    Features = Vectorizer.transform([Article])

    Prediction = Model.predict(Features)[0]
    Margin = Model.decision_function(Features)[0]

    Verdict = "REAL NEWS" if Prediction == 1 else "FAKE NEWS"

    print("\n" + "-" * 60)
    print("ARTICLE:", Name)
    print("PREDICTION:", Verdict)
    print("DECISION SCORE:", round(float(Margin), 4))
    print("FEATURES FOUND:", Features.nnz)

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETED")
print("=" * 60)
