import os
from credentials import HUGGINGFACE_TOKEN, OPENAI_API_KEY

DATASET_PATH = "./data/data.json"
OUTPUT_DIR = "./output"

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LOCAL_MODEL_DIR = "./src/models"  # Directory to save/load the local model
USE_LOCAL_MODEL_STORAGE = True  # Flag to control local model storage

DATA_LIMIT = 100  # Maximum number of dataset items to process
