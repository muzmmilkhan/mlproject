import logging
from datetime import datetime
import os

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[ %(asctime)s ] - %(lineno)d - %(name)s - %(levelname)s - %(message)s',
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
    logging.info("This is an info message.")
    logging.error("This is an error message.")
    logging.warning("This is a warning message.")
    logging.debug("This is a debug message.")
    logging.critical("This is a critical message.")
    
    print(f"Logs are being saved to {LOG_FILE_PATH}")  # For confirmation in console
