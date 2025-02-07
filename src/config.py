import sys
  # we want to import those tokens through config.py so credentials.py is not exposed to the public
from credentials import HUGGINGFACE_TOKEN, OPENAI_API_KEY

DATASET_PATH = "./data/500_question_answers.json"
OUTPUT_DIR = "./output"

ENABLE_DETAILED_LOGS = True

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def is_running_on_colab():
    return "google.colab" in sys.modules


if is_running_on_colab():
    # Path in Google Drive if running on Colab
    LOCAL_MODEL_DIR = "/content/drive/MyDrive/HiddenOracle3/models"
else:
    # Local path on your machine
    LOCAL_MODEL_DIR = "../models"

USE_LOCAL_MODEL_STORAGE = True  # Flag to control local model storage

# Default limit for how many Q&A pairs to process
DEFAULT_DATA_LIMIT = 5

# We keep only the parameters needed:
LAYER_INDEX = 20
TRAIN_TEST_SPLIT_RATIO = 0.8

# If True, skip running LLM and load hidden states+labels from disk
USE_PRECOMPUTED_DATA = True
CACHED_DATA_PATH = "./output/precomputed_data.npz"
