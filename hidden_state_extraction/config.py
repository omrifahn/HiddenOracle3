import sys
from .credentials import HUGGINGFACE_TOKEN, OPENAI_API_KEY

# Input dataset file; ensure triviaqa_data.json is placed in this folder.
DATASET_PATH = "./triviaqa_data.json"
# Output directory for enriched data.
OUTPUT_DIR = "./output"

ENABLE_DETAILED_LOGS = True

LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"


def is_running_on_colab():
    return "google.colab" in sys.modules


if is_running_on_colab():
    LOCAL_MODEL_DIR = "/content/drive/MyDrive/HiddenOracle3/models"
else:
    LOCAL_MODEL_DIR = "../models"

USE_LOCAL_MODEL_STORAGE = True

# Process all samples (set to an integer to limit)
DEFAULT_DATA_LIMIT = 5

LAYER_INDEX = 20
TRAIN_TEST_SPLIT_RATIO = 0.8  # (not used in enrichment)

USE_PRECOMPUTED_DATA = False
CACHED_DATA_PATH = "./output/precomputed_data.npz"
