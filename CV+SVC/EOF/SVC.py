import pandas as pd
import time, pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import re
from typing import Iterable, List

# --- TQDM IMPORT ---
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.base import BaseEstimator, TransformerMixin


# ==============================================================================
# 🔹 BENGALI CLEANER CLASS (INTEGRATED INTO SCRIPT TO ADD TQDM)
# ==============================================================================
# Define constants for cleaning
BN_DIGITS = '০১২৩৪৫৬৭৮۹'
EN_DIGITS = '0123456789'
BN_PUNCT = '।॥‘’“”—–…•'
EXTRA_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
PUNCT_TABLE = str.maketrans({c: ' ' for c in BN_PUNCT + EXTRA_PUNCT})
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.UNICODE)
MENTION_RE = re.compile(r'[@#][\w\-_]+', re.UNICODE)
MULTISPACE_RE = re.compile(r'\s+', re.UNICODE)

def bn_normalize(text: str) -> str:
    if not isinstance(text, str): return ''
    text = text.strip()
    text = URL_RE.sub(' ', text)
    text = MENTION_RE.sub(' ', text)
    text = text.translate(PUNCT_TABLE)
    for bn_digit, en_digit in zip(BN_DIGITS, EN_DIGITS):
        text = text.replace(bn_digit, en_digit)
    text = MULTISPACE_RE.sub(' ', text)
    return text.strip()

class BengaliCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X: Iterable[str], y=None):
        return self
    
    # --- CAREFUL ADDITION OF TQDM HERE ---
    # This will show a progress bar whenever the pipeline cleans text data.
    def transform(self, X: Iterable[str]) -> List[str]:
        return [bn_normalize(x) for x in tqdm(X, desc="Cleaning text")]
# ==============================================================================


# ========== Define Output Directories ==========
RESULTS_DIR = "results"
MODEL_DIR = "model"
PICTURE_DIR = "picture"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PICTURE_DIR, exist_ok=True)

# ========== 1. Load and Clean Data ==========
df = pd.read_csv(r"D:\Abdullah\Research_with_fuad\data\dt1.csv")
df = df.dropna(subset=['text', 'eof']).copy()
df['text'] = df['text'].astype(str).str.strip()
df['eof'] = df['eof'].astype(str).str.strip().str.lower()
df = df[df['eof'] != 'na'].reset_index(drop=True)

X = df['text']
y = df['eof']

# ========== 2. Split ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ========== 3. Pipeline for SVC ==========
pipeline = Pipeline([
    ("cleaner", BengaliCleaner()),
    ("vect", CountVectorizer()),
    ("clf", SVC(probability=True))
])

param_grid = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1, 10],
    "clf__kernel": ['linear', 'rbf'],
    "clf__gamma": ['scale']
}

# ========== 5. Training with Time ==========
print("Starting SVC model training with GridSearchCV...")
start_train = time.time()
# Note: GridSearchCV's `verbose` parameter is its built-in progress bar.
# `verbose=1` shows updates, so we don't wrap grid.fit() with tqdm.
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"Training finished in {train_time:.2f} seconds.")

best_model = grid.best_estimator_
print("\nBest parameters found for SVC:")
print(grid.best_params_)

# ========== 6. Inference with Time ==========
start_infer = time.time()
y_pred = best_model.predict(X_val)
inference_time = time.time() - start_infer

# ========== 7. Metrics ==========
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average="weighted")
recall = recall_score(y_val, y_pred, average="weighted")

auc_score_metric = None
if hasattr(best_model, "predict_proba"):
    y_proba_full = best_model.predict_proba(X_val)
    if len(set(y)) == 2:
        auc_score_metric = roc_auc_score(y_val, y_proba_full[:, 1])
    else:
        auc_score_metric = roc_auc_score(y_val, y_proba_full, multi_class="ovr")

cm = confusion_matrix(y_val, y_pred)

print("\n--- SVC Validation Metrics ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
if auc_score_metric is not None:
    print(f"AUC (OVR): {auc_score_metric:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# ========== 8. Save Metrics ==========
metrics = {
    "accuracy": acc, "f1_score": f1, "recall": recall, "auc": auc_score_metric,
    "train_time": train_time, "inference_time": inference_time, "confusion_matrix": cm.tolist()
}
metrics_path = os.path.join(RESULTS_DIR, "svc_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"\nMetrics saved to: {metrics_path}")

# ========== 9. Save Model and Get Size ==========
model_path = os.path.join(MODEL_DIR, "svc.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

model_size = os.path.getsize(model_path) / 1024
print(f"Model saved to: {model_path}")
print(f"Model Size: {model_size:.2f} KB")

# ========== 10. Confusion Matrix Plot ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix - SVC")
plt.xlabel("Predicted")
plt.ylabel("True")
cm_path = os.path.join(PICTURE_DIR, "svc_cm.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to: {cm_path}")

# ========== 11. Plotting the ROC AUC Curve for the Multiclass Model ==========
if hasattr(best_model, "predict_proba"):
    class_labels = best_model.classes_
    n_classes = len(class_labels)
    y_val_binarized = label_binarize(y_val, classes=class_labels)
    y_pred_proba = best_model.predict_proba(X_val)

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_val_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve - SVC')
    plt.legend(loc="lower right", fontsize=11)
    
    roc_path = os.path.join(PICTURE_DIR, "svc_roc_multiclass.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-class ROC curve plot saved to: {roc_path}")
else:
    print("\nROC curve could not be generated. Ensure SVC has probability=True.")