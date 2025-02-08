import random
import numpy as np
import torch
import os
import json
import time

from .local_llm import load_local_model
from .pipeline import load_dataset, precompute_hidden_states_and_labels
from .config import (
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
    start_run_time = time.time()
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
        # If loading precomputed data, summary_stats might not be available.
        summary_stats = {}
    else:
        print("Loading local model...")
        _generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
        print("Local model loaded successfully.")

        print("Enriching dataset with hidden states and labels...")
        enriched_data, error_log, summary_stats = precompute_hidden_states_and_labels(
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

    total_run_time = time.time() - start_run_time

    # Compute runtime statistics if available.
    if (
        "runtime_per_datapoint" in summary_stats
        and summary_stats["runtime_per_datapoint"]
    ):
        runtimes = summary_stats["runtime_per_datapoint"]
        avg_runtime = sum(runtimes) / len(runtimes)
        min_runtime = min(runtimes)
        max_runtime = max(runtimes)
    else:
        avg_runtime = min_runtime = max_runtime = None

    # Build a detailed run report.
    run_report = {
        "parameters": {
            "DEFAULT_DATA_LIMIT": DEFAULT_DATA_LIMIT,
            "LAYER_INDEX": LAYER_INDEX,
            "USE_PRECOMPUTED_DATA": USE_PRECOMPUTED_DATA,
            "LOCAL_MODEL_NAME": LOCAL_MODEL_NAME,
        },
        "enriched_samples": len(enriched_data),
        "error_count": len(error_log),
        "errors": error_log,
        "evaluation_summary": {
            "total_samples": summary_stats.get("total_samples", 0),
            "factual_string_match": summary_stats.get("factual_string_match", 0),
            "factual_api": summary_stats.get("factual_api", 0),
            "hallucinations": summary_stats.get("hallucinations", 0),
            "runtime_statistics": {
                "average": avg_runtime,
                "minimum": min_runtime,
                "maximum": max_runtime,
            },
        },
        "total_run_time": total_run_time,
    }

    # Save detailed run report as "run_report_part_2.json"
    report_file = os.path.join(OUTPUT_DIR, "run_report_part_2.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=4)
    print(f"Run report saved to '{report_file}'.")


if __name__ == "__main__":
    main()
