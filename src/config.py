import sys
from credentials import (
    HUGGINGFACE_TOKEN,
    OPENAI_API_KEY,
)  # we want to import those tokens through config.py so credentials.py is not exposed to the public

DATASET_PATH = "./data/500_question_answers.json"
OUTPUT_DIR = "./output"

ENABLE_DETAILED_LOGS = True

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


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

# Additional training/inference parameters:
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LAYER_INDEX = 20
TRAIN_TEST_SPLIT_RATIO = 0.8

# TODO OMRI - connect to harder Q&A dataset (because currently, llama 90% right)
