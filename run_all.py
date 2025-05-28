# This is a script to run all the scripts
import os
def run_script(script_name):
    print(f"Running {script_name}...")
    os.system(f"python {script_name}")
if __name__ == "__main__":
    scripts = [
        "result_comparison.py"
        "result_combined.py",
    ]
    
    for script in scripts:
        run_script(script)
    
    print("All scripts executed successfully.")