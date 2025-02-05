import sys
from credentials import (
    HUGGINGFACE_TOKEN,
    OPENAI_API_KEY,
)  # we want to import those tokens through config.py so credentials.py is not exposed to the public

DATASET_PATH = "./data/500_question_answers.json"
OUTPUT_DIR = "./output"

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def is_running_on_colab():
    return "google.colab" in sys.modules


if is_running_on_colab():
    # Path in Google Drive
    LOCAL_MODEL_DIR = "/content/drive/MyDrive/HiddenOracle3/models"
else:
    # Local path on your machine
    LOCAL_MODEL_DIR = "../models"

USE_LOCAL_MODEL_STORAGE = True  # Flag to control local model storage

# Set default DATA_LIMIT here
DEFAULT_DATA_LIMIT = 5
