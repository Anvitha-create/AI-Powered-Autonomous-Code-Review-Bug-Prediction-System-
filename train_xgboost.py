from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import numpy as np

print("=" * 60)
print("Loading CodeXGLUE Defect Detection Dataset...")
print("=" * 60)

# Load dataset
dataset = load_dataset("google/code_x_glue_cc_defect_detection")

# Extract code and labels
train_codes = dataset["train"]["func"]
train_labels = np.array(dataset["train"]["target"], dtype=np.int32)

test_codes = dataset["test"]["func"]
test_labels = np.array(dataset["test"]["target"], dtype=np.int32)

print(f"Training samples : {len(train_codes)}")
print(f"Testing samples  : {len(test_codes)}")

print("\nConverting code to TF-IDF vectors...")

# TF-IDF Features
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_codes)
X_test = vectorizer.transform(test_codes)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print("Feature matrix shape:", X_train.shape)

print("\nTraining XGBoost model...")

# XGBoost Model
model = XGBClassifier(
    objective="binary:logistic",
    max_depth=7,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

model.fit(X_train, train_labels)

print("Training Complete!")

print("\nRunning Predictions...")

predictions = model.predict(X_test)

accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

cm = confusion_matrix(test_labels, predictions)

print("\nConfusion Matrix")
print(cm)

print("\nSaving model...")

model.save_model("xgboost_defect_detector.json")

print("Model saved as:")
print("xgboost_defect_detector.json")

print("\nDone!")