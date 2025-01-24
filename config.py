import openai

# ---------------------------------------
# Configuration Variables
# ---------------------------------------

# (1) OpenAI API Key
OPENAI_API_KEY = "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"

# (2) Paths
DATASET_PATH = "./data/data.json"

# (3) Local model name
LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"


# ---------------------------------------
# OpenAI Setup
# ---------------------------------------
def configure_openai() -> None:
    """
    Sets the OpenAI API key.
    Call this once before using any OpenAI endpoints.
    """
    openai.api_key = OPENAI_API_KEY
