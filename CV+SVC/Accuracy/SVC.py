import pandas as pd
import time, pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC # <-- IMPORT SVC
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
# --- IMPORTS REQUIRED FOR MULTI-CLASS ROC PLOTTING ---
from sklearn.preprocessing import label_binarize
from itertools import cycle

# 🔹 Import cleaner (ensure text_cleaning.py is in the same directory)
from text_cleaning import BengaliCleaner

# ========== Define Output Directories ==========
# Define paths for all outputs
RESULTS_DIR = "results"
MODEL_DIR = "model"
PICTURE_DIR = "picture"

# Create directories if they don't already exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PICTURE_DIR, exist_ok=True)

# ========== 1. Load and Clean Data ==========
df = pd.read_csv(r"D:\Abdullah\Research_with_fuad\data\dt1.csv")
# Drop rows where 'text' or 'accuracy' are missing
df = df.dropna(subset=['text', 'accuracy']).copy()

# Clean the 'text' and 'accuracy' columns
df['text'] = df['text'].astype(str).str.strip()
df['accuracy'] = df['accuracy'].astype(str).str.strip().str.lower()

# IMPORTANT: Filter out rows where 'accuracy' is 'na'
df = df[df['accuracy'] != 'na'].reset_index(drop=True)

X = df['text']
y = df['accuracy']

# ========== 2. Split ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ========== 3. Pipeline for SVC ==========
pipeline = Pipeline([
    ("cleaner", BengaliCleaner()),
    ("vect", CountVectorizer()),
    # Use SVC. probability=True is required for ROC curve but makes training slower.
    ("clf", SVC(probability=True))
])

# Define a new parameter grid for SVC
param_grid = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1, 10],            # Regularization parameter
    "clf__kernel": ['linear', 'rbf'],  # Type of kernel
    "clf__gamma": ['scale']            # Kernel coefficient for 'rbf'
}

# ========== 5. Training with Time ==========
print("Starting SVC model training with GridSearchCV...")
start_train = time.time()
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

# Correctly handle ROC/AUC score for the metrics file
auc_score_metric = None
if hasattr(best_model, "predict_proba"):
    y_proba_full = best_model.predict_proba(X_val)
    if len(set(y)) == 2:  # Binary classification
        auc_score_metric = roc_auc_score(y_val, y_proba_full[:, 1])
    else:  # Multi-class classification
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
    "accuracy": acc,
    "f1_score": f1,
    "recall": recall,
    "auc": auc_score_metric,
    "train_time": train_time,
    "inference_time": inference_time,
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(RESULTS_DIR, "svc_metrics.csv") # <-- CHANGED FILENAME
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"\nMetrics saved to: {metrics_path}")

# ========== 9. Save Model and Get Size ==========
model_path = os.path.join(MODEL_DIR, "svc.pkl") # <-- CHANGED FILENAME
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

model_size = os.path.getsize(model_path) / 1024
print(f"Model saved to: {model_path}")
print(f"Model Size: {model_size:.2f} KB")

# ========== 10. Confusion Matrix Plot ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix - SVC") # <-- CHANGED TITLE
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = os.path.join(PICTURE_DIR, "svc_cm.png") # <-- CHANGED FILENAME
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to: {cm_path}")


# ========== 11. Plotting the ROC AUC Curve for the Multiclass Model ==========

if hasattr(best_model, "predict_proba"):
    # --- 1. Get class labels and binarize the true labels ---
    class_labels = best_model.classes_
    n_classes = len(class_labels)
    y_val_binarized = label_binarize(y_val, classes=class_labels)

    # --- 2. Get prediction probabilities ---
    y_pred_proba = best_model.predict_proba(X_val)

    # --- 3. Compute ROC curve and ROC area for each class ---
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # --- 4. Compute micro-average ROC curve and ROC area ---
    fpr["micro"], tpr["micro"], _ = roc_curve(y_val_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # --- 5. Plot all ROC curves ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    # --- 6. Formatting and Saving the plot ---
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve - SVC') # <-- CHANGED TITLE
    plt.legend(loc="lower right", fontsize=11)
    
    roc_path = os.path.join(PICTURE_DIR, "svc_roc_multiclass.png") # <-- CHANGED FILENAME
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Multi-class ROC curve plot saved to: {roc_path}")
else:
    print("\nROC curve could not be generated. Ensure SVC has probability=True.")