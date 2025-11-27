import torch
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate


MODEL_NAME = "distilroberta-base"  # Faster model
OUTPUT_DIR = "trained_emotion_model"
NUM_LABELS = 28  # Correct number of GoEmotions labels

dataset = load_dataset("go_emotions")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------------------------
# Multi-hot label encoding
# ---------------------------
def tokenize(batch):
    encoding = tokenizer(batch["text"], padding=True, truncation=True)
    labels_list = batch["labels"]

    encoding["labels"] = []
    for labels in labels_list:
        vec = np.zeros(NUM_LABELS, dtype=np.float32)
        for lbl in labels:
            if lbl < NUM_LABELS:
                vec[lbl] = 1.0
        encoding["labels"].append(vec.tolist())

    return encoding


tokenized = dataset.map(tokenize, batched=True)


# ---------------------------
# Model
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)
model.to(device)


# ---------------------------
# Metrics
# ---------------------------
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    logits = torch.tensor(logits).to(device)
    labels = torch.tensor(labels).to(device)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

    # Exact match accuracy
    exact_match = (preds == labels.int()).all(dim=1).float().mean().item()

    # Convert to CPU + numpy for sklearn
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Multi-label macro F1
    f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    return {
        "exact_match_accuracy": exact_match,
        "f1": f1,
    }


# ---------------------------
# Float label collator
# ---------------------------
class FloatDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["labels"].float()
        return batch


collator = FloatDataCollator(tokenizer)


# ---------------------------
# Fast GPU Training Settings
# ---------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,  # RTX 3050 supports FP16
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    label_names=["labels"],
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)


# ---------------------------
# Use subset for faster training
# ---------------------------
trainer.train_dataset = trainer.train_dataset.select(range(20000))
trainer.eval_dataset = trainer.eval_dataset.select(range(2000))

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("🎉 Training done on GPU! Saved to trained_emotion_model/")
