import subprocess
import sys

# Use the same Python interpreter that's running main.py (inside venv)
python_exec = sys.executable

# --- Step 1: Generate embeddings ---
subprocess.run([python_exec, "generate_embeddings_res.py"])
subprocess.run([python_exec, "generate_embeddings_acc.py"])

# --- Step 2: Train Logistic Regression Models ---
subprocess.run([python_exec, "train_lr_accuracy.py"])
subprocess.run([python_exec, "train_lr_eof.py"])
subprocess.run([python_exec, "train_lr_responsiveness.py"])
