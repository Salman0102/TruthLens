
import numpy as np
import joblib
import time

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
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
print("FAKE NEWS DETECTION - MACHINE LEARNING MODEL TRAINING")
print("=" * 65)

# STEP 1: Load extracted features

print("\nSTEP 1: Loading TF-IDF features...")

XTrain = load_npz("models/X_train.npz")
XTest = load_npz("models/X_test.npz")

YTrain = np.load("models/y_train.npy")
YTest = np.load("models/y_test.npy")

print("Training data:", XTrain.shape)
print("Testing data:", XTest.shape)

# STEP 2: Initialize models

Models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42
    ),

    "Naive Bayes": MultinomialNB(
        alpha=0.1
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        random_state=42
    )
}

Results = []

# STEP 3: Train and evaluate models

for Name, Model in Models.items():

    print("\n" + "=" * 65)
    print("TRAINING:", Name)
    print("=" * 65)

    StartTime = time.time()

    Model.fit(XTrain, YTrain)

    TrainingTime = time.time() - StartTime

    print("Training completed!")
    print("Training time:", round(TrainingTime, 2), "seconds")

    # Predict test data
    Predictions = Model.predict(XTest)

    # Calculate evaluation metrics
    Accuracy = accuracy_score(YTest, Predictions)

    Precision = precision_score(
        YTest, Predictions, zero_division=0
    )

    Recall = recall_score(
        YTest, Predictions, zero_division=0
    )

    F1 = f1_score(
        YTest, Predictions, zero_division=0
    )

    print("\nMODEL PERFORMANCE")
    print("Accuracy:", round(Accuracy * 100, 2), "%")
    print("Precision:", round(Precision * 100, 2), "%")
    print("Recall:", round(Recall * 100, 2), "%")
    print("F1 Score:", round(F1 * 100, 2), "%")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            YTest,
            Predictions,
            target_names=["Fake News", "Real News"],
            zero_division=0
        )
    )

    print("CONFUSION MATRIX")
    print(confusion_matrix(YTest, Predictions))

    # Save model
    Filename = Name.lower().replace(" ", "_")
    joblib.dump(Model, "models/" + Filename + ".pkl")

    Results.append({
        "Model": Name,
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Recall": Recall,
        "F1 Score": F1,
        "Training Time": TrainingTime
    })

# STEP 4: Compare all models

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

ResultsData = sorted(
    Results,
    key=lambda X: X["F1 Score"],
    reverse=True
)

for Result in ResultsData:

    print("\nModel:", Result["Model"])
    print("Accuracy:", round(Result["Accuracy"] * 100, 2), "%")
    print("Precision:", round(Result["Precision"] * 100, 2), "%")
    print("Recall:", round(Result["Recall"] * 100, 2), "%")
    print("F1 Score:", round(Result["F1 Score"] * 100, 2), "%")
    print("Training Time:", round(Result["Training Time"], 2), "seconds")

# STEP 5: Save best model

BestModel = ResultsData[0]

BestFilename = (
    BestModel["Model"].lower().replace(" ", "_") + ".pkl"
)

joblib.dump(
    joblib.load("models/" + BestFilename),
    "models/best_model.pkl"
)

print("\n" + "=" * 65)
print("BEST MODEL:", BestModel["Model"])
print("BEST F1 SCORE:", round(BestModel["F1 Score"] * 100, 2), "%")
print("Saved as: models/best_model.pkl")
print("=" * 65)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
