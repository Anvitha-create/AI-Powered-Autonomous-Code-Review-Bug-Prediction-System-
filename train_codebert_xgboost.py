from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Loading CodeXGLUE Dataset...")
print("=" * 60)

dataset = load_dataset("google/code_x_glue_cc_defect_detection")

train_codes = dataset["train"]["func"][:1000]
train_labels = np.array(dataset["train"]["target"][:1000])

test_codes = dataset["test"]["func"][:200]
test_labels = np.array(dataset["test"]["target"][:200])

print(f"Train Samples: {len(train_codes)}")
print(f"Test Samples : {len(test_codes)}")

print("\nLoading CodeBERT...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

model.to(DEVICE)
model.eval()

def generate_embeddings(codes, batch_size=16):
    embeddings = []

    total_batches = (len(codes) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(codes), batch_size):

            batch = codes[i:i + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)

            batch_embeddings = outputs.last_hidden_state.mean(dim=1)

            embeddings.extend(
                batch_embeddings.cpu().numpy()
            )

            current_batch = i // batch_size + 1

            print(
                f"Processed batch {current_batch}/{total_batches}",
                end="\r"
            )

    print()
    return np.array(embeddings)

print("\nGenerating Train Embeddings...")
X_train = generate_embeddings(train_codes)

print("\nGenerating Test Embeddings...")
X_test = generate_embeddings(test_codes)

print("\nTraining XGBoost...")

clf = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

clf.fit(X_train, train_labels)

print("Training Complete!")

print("\nRunning Evaluation...")

preds = clf.predict(X_test)

accuracy = accuracy_score(test_labels, preds)
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)
f1 = f1_score(test_labels, preds)

print("\n" + "=" * 60)
print("CODEBERT + XGBOOST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nSaving model...")

clf.save_model("codebert_xgboost.json")

print("Saved: codebert_xgboost.json")