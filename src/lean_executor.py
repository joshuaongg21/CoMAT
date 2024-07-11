import subprocess
import os

def run_lean(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        return f"LEAN execution failed with error:\n{result.stderr}"
    
    return result.stdout.strip()

def identify_inconsistencies(file_path):
    result = subprocess.run(['lean', file_path], capture_output=True, text=True)
    errors = result.stderr.strip().split('\n')
    return [err for err in errors if err.strip()]