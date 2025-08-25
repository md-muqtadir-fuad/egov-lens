import subprocess
import sys
import time

def run_script(script_name, python_executable):
    """
    A helper function to run a python script and print its status.
    
    Args:
        script_name (str): The name of the python script to run.
        python_executable (str): The path to the python interpreter.
    """
    print(f"\n{'='*20} RUNNING SCRIPT: {script_name} {'='*20}")
    start_time = time.time()
    
    try:
        # Use check=True to raise an error if the script fails (returns a non-zero exit code).
        # This will stop the main script if any sub-script has a problem.
        subprocess.run([python_executable, script_name], check=True)
        end_time = time.time()
        print(f"\n{'-'*20} FINISHED: {script_name} in {end_time - start_time:.2f} seconds {'-'*20}")
    
    except FileNotFoundError:
        print(f"\n[ERROR] The script '{script_name}' was not found. Skipping.")
    
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] The script '{script_name}' failed with exit code {e.returncode}.")
        # Stop the entire process if one script fails
        sys.exit(f"Stopping execution due to failure in {script_name}.")
        
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred while running '{script_name}': {e}")
        sys.exit("Stopping execution.")


def main():
    """
    Orchestrates the running of all embedding and training scripts in the correct order.
    """
    # Use the same Python interpreter that's running main.py (ensures venv is used)
    python_exec = sys.executable
    print(f"Using Python interpreter located at: {python_exec}")

    # Define the sequence of scripts to run
    scripts_to_run = [
     
        # --- STAGE 2: TRAIN ALL MODELS USING THE EMBEDDINGS ---
        "train_lr_eof.py",
        "train_lr_responsiveness.py",
        "train_lr_accuracy.py"
    ]
    
    total_start_time = time.time()
    
    print(f"\nStarting the full pipeline of {len(scripts_to_run)} scripts...")
    
    # Loop through and execute each script
    for script in scripts_to_run:
        run_script(script, python_exec)
        
    total_end_time = time.time()
    
    print(f"\n{'='*20} ALL SCRIPTS COMPLETED SUCCESSFULLY {'='*20}")
    print(f"Total pipeline execution time: {total_end_time - total_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()