import os
import sys
from .credentials import HUGGINGFACE_TOKEN, OPENAI_API_KEY

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input dataset file; ensure triviaqa_data.json is placed in the same folder as this config file.
DATASET_PATH = os.path.join(CURRENT_DIR, "triviaqa_data.json")

# Output directory for enriched data.
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")

ENABLE_DETAILED_LOGS = True

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def is_running_on_colab():
    return "google.colab" in sys.modules


if is_running_on_colab():
    LOCAL_MODEL_DIR = "/content/drive/MyDrive/HiddenOracle3/models"
else:
    LOCAL_MODEL_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")

USE_LOCAL_MODEL_STORAGE = True

# Process all samples (set to an integer to limit)
DEFAULT_DATA_LIMIT = None

LAYER_INDEX = 20
TRAIN_TEST_SPLIT_RATIO = 0.8  # (not used in enrichment)

USE_PRECOMPUTED_DATA = False
CACHED_DATA_PATH = os.path.join(OUTPUT_DIR, "precomputed_data.npz")
