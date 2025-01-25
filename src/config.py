import os

DATASET_PATH = "./data/data.json"
OUTPUT_DIR = "./output"

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
LOCAL_MODEL_DIR = "./models"  # Directory to save/load the local model
USE_LOCAL_MODEL_STORAGE = True  # Flag to control local model storage

# Load API keys from environment variables for security
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
