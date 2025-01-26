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

DATA_LIMIT = 3  # Maximum number of dataset items to process



# TODO restruct and put get back main.py to work
# TODO compare with prev project and improve the config.py file
# TODO mske sure we have closed clean interface  for the 4 options - good green, bad green, bad red, good red