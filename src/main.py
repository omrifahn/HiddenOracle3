import sys
import random
import numpy as np
import torch
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from config import (
    DATASET_PATH,
    LOCAL_MODEL_NAME,
    DEFAULT_DATA_LIMIT,
    OUTPUT_DIR,
    LAYER_INDEX,
    TRAIN_TEST_SPLIT_RATIO,
    USE_PRECOMPUTED_DATA,
    CACHED_DATA_PATH,
)
from pipeline import (
    load_dataset,
    precompute_hidden_states_and_labels,
)
from local_llm import (
    load_local_model,
)

if __name__ == "__main__":
    # Seed everything for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Determine data_limit from command line (if provided)
    data_limit = DEFAULT_DATA_LIMIT
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1.lower() == "none":
            data_limit = None
        else:
            try:
                data_limit = int(arg1)
            except ValueError:
                print(
                    f"Invalid data_limit '{arg1}' provided. Using default {data_limit}."
                )

    # Diagnostic info
    print("\n--- Diagnostic Info ---")
    print(f"USE_PRECOMPUTED_DATA = {USE_PRECOMPUTED_DATA}")
    print(f"CACHED_DATA_PATH     = {CACHED_DATA_PATH}")
    cache_exists = os.path.isfile(CACHED_DATA_PATH)
    print(f"Does the cache file exist? {cache_exists}\n")

    # 1) Load raw dataset
    dataset = load_dataset(DATASET_PATH, data_limit)

    # 2) Process dataset to add new fields (LLM answer, evaluation, hidden vector, label)
    updated_data_file = os.path.join(OUTPUT_DIR, "updated_data.json")
    error_log = []  # initialize error log
    if USE_PRECOMPUTED_DATA and os.path.isfile(updated_data_file):
        print(f"INFO: Loading precomputed updated data from '{updated_data_file}'...")
        with open(updated_data_file, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        print("INFO: Updated data loaded from file.")
    else:
        print("Loading local model now...")
        generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
        print("Local model loaded successfully.")

        print(
            "Enriching dataset with new fields (LLM answer, evaluation, hidden vector, label)..."
        )
        updated_data, error_log = precompute_hidden_states_and_labels(
            samples=dataset, model=model, tokenizer=tokenizer, layer_index=LAYER_INDEX
        )
        print("Dataset enrichment complete.")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(updated_data_file, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=4)
        print(f"Updated data saved to '{updated_data_file}'.")

    # Optionally, extract features and labels for further analysis
    features = []
    labels = []
    for item in updated_data:
        features.append(item["hidden_vector"])
        labels.append(item["label"])
    features_np = np.array(features)
    labels_np = np.array(labels)

    # 3) Train/test split
    train_features, test_features, train_truths, test_truths = train_test_split(
        features_np,
        labels_np,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO,
        random_state=seed,
        shuffle=True,
    )

    # Data balance prints
    train_factual_count = np.sum(train_truths == 0)
    train_halluc_count = np.sum(train_truths == 1)
    test_factual_count = np.sum(test_truths == 0)
    test_halluc_count = np.sum(test_truths == 1)

    print("\nData Balance:")
    print(f"  TRAIN set total: {len(train_truths)}")
    print(f"    Factual (0): {train_factual_count}")
    print(f"    Hallucinating (1): {train_halluc_count}")
    print(f"  TEST set total: {len(test_truths)}")
    print(f"    Factual (0): {test_factual_count}")
    print(f"    Hallucinating (1): {test_halluc_count}")

    # 4) Train logistic regression
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_truths)

    # 5) Evaluate
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_truths, predictions)
    print(f"\nLogistic Regression Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(test_truths, predictions, labels=[0, 1])
    good_green = cm[0, 0]
    bad_green = cm[1, 0]
    bad_red = cm[0, 1]
    good_red = cm[1, 1]
    print("\nConfusion Matrix (Factual/Hallucinating vs. Green/Red):")
    print(f"  Good Green (Factual & Green): {good_green}")
    print(f"  Bad Green (Hallucinating & Green): {bad_green}")
    print(f"  Bad Red (Factual & Red): {bad_red}")
    print(f"  Good Red (Hallucinating & Red): {good_red}")

    # Write run report
    run_report = {
        "parameters": {
            "data_limit": data_limit,
            "LAYER_INDEX": LAYER_INDEX,
            "TRAIN_TEST_SPLIT_RATIO": TRAIN_TEST_SPLIT_RATIO,
            "USE_PRECOMPUTED_DATA": USE_PRECOMPUTED_DATA,
        },
        "data_balance": {
            "train_total": int(len(train_truths)),
            "train_factual_count": int(train_factual_count),
            "train_halluc_count": int(train_halluc_count),
            "test_total": int(len(test_truths)),
            "test_factual_count": int(test_factual_count),
            "test_halluc_count": int(test_halluc_count),
        },
        "evaluation": {
            "accuracy": float(accuracy),
            "confusion_matrix": {
                "good_green": int(good_green),
                "bad_green": int(bad_green),
                "bad_red": int(bad_red),
                "good_red": int(good_red),
            },
        },
        "errors": error_log,
    }
    report_file_path = os.path.join(OUTPUT_DIR, "run_report.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=4)
    print(f"Run report saved to '{report_file_path}'.")
