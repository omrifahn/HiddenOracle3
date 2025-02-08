import random
import numpy as np
import torch
import os
import json

from local_llm import load_local_model
from pipeline import load_dataset, precompute_hidden_states_and_labels
from config import (
    DATASET_PATH,
    LOCAL_MODEL_NAME,
    DEFAULT_DATA_LIMIT,
    OUTPUT_DIR,
    LAYER_INDEX,
    USE_PRECOMPUTED_DATA,
)

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
random.seed(seed)


def main():
    print("Loading dataset from:", DATASET_PATH)
    dataset = load_dataset(DATASET_PATH, DEFAULT_DATA_LIMIT)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    enriched_data_file = os.path.join(OUTPUT_DIR, "hidden_state_data.json")
    error_log = []

    if USE_PRECOMPUTED_DATA and os.path.isfile(enriched_data_file):
        print(f"INFO: Loading precomputed enriched data from '{enriched_data_file}'...")
        with open(enriched_data_file, "r", encoding="utf-8") as f:
            enriched_data = json.load(f)
        print("INFO: Enriched data loaded from file.")
    else:
        print("Loading local model...")
        _generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
        print("Local model loaded successfully.")

        print("Enriching dataset with hidden states and labels...")
        enriched_data, error_log = precompute_hidden_states_and_labels(
            dataset, model, tokenizer, layer_index=LAYER_INDEX
        )
        print("Dataset enrichment complete.")

        with open(enriched_data_file, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, ensure_ascii=False, indent=4)
        print(f"Enriched data saved to '{enriched_data_file}'.")

    if error_log:
        print("Errors encountered during processing:")
        for error in error_log:
            print(error)


if __name__ == "__main__":
    main()
