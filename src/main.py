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

    # Figure out data_limit from command line (if given)
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

    # 1) Load raw dataset
    dataset = load_dataset(DATASET_PATH, data_limit)

    # 2) Either load precomputed hidden states/labels or run LLM
    if USE_PRECOMPUTED_DATA and os.path.isfile(CACHED_DATA_PATH):
        print(f"Loading precomputed features and labels from {CACHED_DATA_PATH}...")
        cached_data = np.load(CACHED_DATA_PATH, allow_pickle=True)
        features = cached_data["features"]
        # LLaMA ground-truth labels: 0 => factual, 1 => hallucinating
        llama_truth_labels = cached_data["labels"]
        result_data = cached_data["result_data"].tolist()
        print("Features and labels loaded from cache.")
    else:
        print("Loading local model...")
        generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
        print("Local model loaded.")

        print("Precomputing hidden states and labels...")
        # returns: features, llama_truth_labels, result_data
        features, llama_truth_labels, result_data = precompute_hidden_states_and_labels(
            samples=dataset, model=model, tokenizer=tokenizer, layer_index=LAYER_INDEX
        )
        print("Precomputation complete.")

        # (Optional) save result_data to JSON for inspection
        output_file_path = os.path.join(OUTPUT_DIR, "output_data.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"Detailed results saved to '{output_file_path}'.")

        # Save precomputed data to .npz
        print(f"Saving precomputed features and labels to {CACHED_DATA_PATH}...")
        np.savez(
            CACHED_DATA_PATH,
            features=features,
            labels=llama_truth_labels,
            result_data=np.array(result_data, dtype=object),
        )
        print("Cache saved.")

    # 3) Train/test split
    (
        train_features,
        test_features,
        train_truths,
        test_truths,
    ) = train_test_split(
        features,
        llama_truth_labels,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO,
        random_state=seed,
        shuffle=True,
    )

    # ---- Print out the data balance (factual vs. hallucinating) in TRAIN set and TEST set ----
    train_factual_count = np.sum(train_truths == 0)
    train_halluc_count = np.sum(train_truths == 1)
    test_factual_count = np.sum(test_truths == 0)
    test_halluc_count = np.sum(test_truths == 1)

    print("\nData Balance:")
    print(f"  TRAIN set total: {len(train_truths)}")
    print(f"    Factual (0)     : {train_factual_count}")
    print(f"    Hallucinating (1): {train_halluc_count}")

    print(f"  TEST set total : {len(test_truths)}")
    print(f"    Factual (0)     : {test_factual_count}")
    print(f"    Hallucinating (1): {test_halluc_count}")

    # 4) Train logistic regression
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_truths)

    # 5) Evaluate
    red_green_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_truths, red_green_predictions)
    print(f"\nLogistic Regression Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Build confusion matrix in new terms
    # test_truths (rows): 0 => LLaMA factual, 1 => LLaMA hallucinating
    # predictions (cols): 0 => "Green", 1 => "Red"
    cm = confusion_matrix(test_truths, red_green_predictions)

    # cm = [[TN, FP],
    #       [FN, TP]]
    # Let's rename them:
    good_green = cm[0, 0]  # factual & green
    bad_green = cm[1, 0]  # hallucinating & green
    bad_red = cm[0, 1]  # factual & red
    good_red = cm[1, 1]  # hallucinating & red

    print("\nConfusion Matrix (Factual/Hallucinating vs. Green/Red):")
    print(f"  Good Green (Factual & Green)         : {good_green}")
    print(f"  Bad Green  (Hallucinating & Green)    : {bad_green}")
    print(f"  Bad Red    (Factual & Red)            : {bad_red}")
    print(f"  Good Red   (Hallucinating & Red)      : {good_red}")

    #  SAVE RUN REPORT TO JSON
    run_report = {
        "parameters": {
            "data_limit": data_limit,
            "LAYER_INDEX": LAYER_INDEX,
            "TRAIN_TEST_SPLIT_RATIO": TRAIN_TEST_SPLIT_RATIO,
            "USE_PRECOMPUTED_DATA": USE_PRECOMPUTED_DATA,
        },
        "data_balance": {
            "train_total": len(train_truths),
            "train_factual_count": int(train_factual_count),
            "train_halluc_count": int(train_halluc_count),
            "test_total": len(test_truths),
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
    }

    report_file_path = os.path.join(OUTPUT_DIR, "run_report.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=4)
    print(f"\nRun report saved to '{report_file_path}'.")
