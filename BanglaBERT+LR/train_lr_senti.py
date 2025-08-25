# train_lr_sentiment.py

import pandas as pd
import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from itertools import cycle

# --- MODIFICATION 1: Import LogisticRegression ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, uniform
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ========== Define Input and Output Directories ==========
EMBEDDINGS_PATH = 'embeddings_sentiment.npy'
LABELS_PATH = 'labels_sentiment.csv'
# This script is dedicated to sentiment analysis
TARGET_LABEL = 'sentiment'

RESULTS_DIR = "results"
MODEL_DIR = "model"
PICTURE_DIR = "picture"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PICTURE_DIR, exist_ok=True)


def main():
    """
    Main function to train and evaluate a Logistic Regression model for sentiment analysis.
    """
    # 1. Load Embeddings and Labels
    print("Loading pre-generated sentiment embeddings and labels...")
    X = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    y = labels_df[TARGET_LABEL]

    print(f"Data loaded. Found {len(X)} samples.")
    print("Class distribution:\n", y.value_counts())

    # 2. Split the Data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # --- MODIFICATION 2: Define Model and Hyperparameters for Logistic Regression ---
    # Increasing max_iter helps prevent convergence warnings.
    lr = LogisticRegression(
        random_state=42,
        solver='lbfgs', # Using a faster solver
        max_iter=1000, # Sufficient for 'lbfgs' to converge
    )

    # Define the hyperparameter distributions to sample from
    param_dist = {
        # loguniform is efficient for searching regularization hyperparameters.
        'C': loguniform(0.01, 100),
    }

    # 4. Training with RandomizedSearchCV
    print(f"\nStarting RandomizedSearchCV for '{TARGET_LABEL}' with Logistic Regression...")
    start_train = time.time()
    
    # --- MODIFICATION 3: Use the Logistic Regression model in the search ---
    random_search = RandomizedSearchCV(
        lr,  # Use the Logistic Regression model instance
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training finished in {train_time:.2f} seconds.")

    best_model = random_search.best_estimator_
    print("\nBest parameters found:", random_search.best_params_)

    # 5. Evaluation (No changes needed in logic)
    start_infer = time.time()
    y_pred = best_model.predict(X_val)
    inference_time = time.time() - start_infer

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")

    auc_score_metric = None
    y_proba_full = best_model.predict_proba(X_val)
    if len(y.unique()) == 2:
        auc_score_metric = roc_auc_score(y_val, y_proba_full[:, 1])
    else:
        auc_score_metric = roc_auc_score(
            y_val, y_proba_full, multi_class="ovr")

    print(f"\n--- Validation Metrics for '{TARGET_LABEL}' ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC (OVR): {auc_score_metric:.4f}")
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

    # --- MODIFICATION 4: Update artifact filenames and plot titles ---

    # Save Metrics
    metrics = {"target_label": TARGET_LABEL, "accuracy": acc, "f1_score": f1, "recall": recall,
               "auc": auc_score_metric, "train_time": train_time, "inference_time": inference_time}
    metrics_path = os.path.join(RESULTS_DIR, f"lr_metrics_{TARGET_LABEL}.csv") # lr_
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    # Save Model
    model_path = os.path.join(MODEL_DIR, f"lr_model_{TARGET_LABEL}.pkl") # lr_
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to: {model_path}")

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title(f"Confusion Matrix - Logistic Regression for '{TARGET_LABEL}'") # Updated title
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(PICTURE_DIR, f"lr_cm_{TARGET_LABEL}.png") # lr_
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    # Save ROC Curve Plot
    class_labels = best_model.classes_
    y_val_binarized = label_binarize(y_val, classes=class_labels)
    n_classes = len(class_labels)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(
            y_val_binarized[:, i], y_proba_full[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_val_binarized.ravel(), y_proba_full.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC (area = {roc_auc["micro"]:0.2f})', color='deeppink', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC of class {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Multi-class ROC Curve - Logistic Regression for '{TARGET_LABEL}'") # Updated title
    plt.legend(loc="lower right")
    roc_path = os.path.join(PICTURE_DIR, f"lr_roc_{TARGET_LABEL}.png") # lr_
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve plot saved to: {roc_path}")


if __name__ == '__main__':
    main()