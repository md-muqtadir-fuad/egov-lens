import pandas as pd
import time, pickle, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc # <-- Added 'auc'
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

# ========== 1. Load Data ==========
df = pd.read_csv(r"D:\Abdullah\Research_with_fuad\data\dt1.csv")
X = df["text"]
y = df["sentiment"].astype(str).str.strip()

# ========== 2. Split ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ========== 3. Pipeline ==========
pipeline = Pipeline([
    ("cleaner", BengaliCleaner()),
    ("vect", CountVectorizer()),
    ("clf", LogisticRegression(max_iter=500))
])

param_grid = {
    "vect__ngram_range": [(1, 1), (1, 2)],
    "vect__max_df": [0.9, 1.0],
    "vect__min_df": [1, 2],
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["liblinear"]
}

# ========== 5. Training with Time ==========
print("Starting model training with GridSearchCV...")
start_train = time.time()
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
train_time = time.time() - start_train
print(f"Training finished in {train_time:.2f} seconds.")

best_model = grid.best_estimator_

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

print("\n--- Validation Metrics ---")
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

metrics_path = os.path.join(RESULTS_DIR, "logistic_regression_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"\nMetrics saved to: {metrics_path}")

# ========== 9. Save Model and Get Size ==========
model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

model_size = os.path.getsize(model_path) / 1024
print(f"Model saved to: {model_path}")
print(f"Model Size: {model_size:.2f} KB")

# ========== 10. Confusion Matrix Plot ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = os.path.join(PICTURE_DIR, "logistic_regression_cm.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to: {cm_path}")


# ========== 11. Plotting the ROC AUC Curve for the Multiclass Model ==========

# This code runs for both binary and multi-class cases
if hasattr(best_model, "predict_proba"):

    # --- 1. Get class labels and binarize the true labels ---
    class_labels = best_model.classes_
    n_classes = len(class_labels)
    y_val_binarized = label_binarize(y_val, classes=class_labels)

    # --- 2. Get prediction probabilities for the validation set ---
    y_pred_proba = best_model.predict_proba(X_val)

    # --- 3. Compute ROC curve and ROC area for each class ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # --- 4. Compute micro-average ROC curve and ROC area ---
    fpr["micro"], tpr["micro"], _ = roc_curve(y_val_binarized.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # --- 5. Plot all ROC curves ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    # Plot the micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    # Plot the ROC curve for each class
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
    plt.title('Multi-class Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    
    roc_path = os.path.join(PICTURE_DIR, "logistic_regression_roc_multiclass.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free up memory

    print(f"Multi-class ROC curve plot saved to: {roc_path}")

else:
    print("\nROC curve could not be generated because the model does not have a 'predict_proba' method.")