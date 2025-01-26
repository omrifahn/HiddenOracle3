import os
from credentials import (
    HUGGINGFACE_TOKEN,
    OPENAI_API_KEY,
)  # we want to import those tokens through config.py so credentials.py is not exposed to the public

DATASET_PATH = "./data/500_question_answers"
OUTPUT_DIR = "./output"

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LOCAL_MODEL_DIR = "./src/models"  # Directory to save/load the local model
USE_LOCAL_MODEL_STORAGE = True  # Flag to control local model storage

DATA_LIMIT = 100  # Maximum number of dataset items to process
