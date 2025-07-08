import sys
import os
import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Saves a Python object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists
        logging.info(f"Saving object to {file_path}")
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
