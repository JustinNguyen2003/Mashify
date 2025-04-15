import os
import shutil
import re

def clear_directory(path):
    """Takes a path to a directory and clears it."""
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove folder and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def get_all_filepaths_sorted(directory):
    """Returns all the full filepaths of the files in a directory."""
    def extract_number(f):
        match = re.search(r'\d+', f)
        return int(match.group()) if match else -1
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort(key=extract_number)
    return [os.path.join(directory, f) for f in files]