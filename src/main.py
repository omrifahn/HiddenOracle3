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
        # Ground-truth states: 0 => factual, 1 => hallucinating
        llama_truth_labels = cached_data["labels"]
        result_data = cached_data["result_data"].tolist()
        print("Features and labels loaded from cache.")
    else:
        print("Loading local model...")
        generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
        print("Local model loaded.")

        print("Precomputing hidden states and labels...")
        # returns: all_features_np, all_labels_np, result_data
        # all_labels_np = 0 for factual, 1 for hallucinating
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

    # 4) Train logistic regression
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_truths)

    # 5) Evaluate
    red_green_predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_truths, red_green_predictions)
    print(f"\nLogistic Regression Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Build confusion matrix in the new terms
    # test_truths (rows):    0 => LLaMA factual, 1 => LLaMA hallucinating
    # red_green_predictions (cols): 0 => classifier says "Green", 1 => classifier says "Red"
    cm = confusion_matrix(test_truths, red_green_predictions)

    # Confusion matrix is typically:
    # [[TN, FP],
    #  [FN, TP]]
    # But let's rename them in the new terms:
    good_green = cm[0, 0]  # factual & green
    bad_green = cm[1, 0]  # hallucinating & green
    bad_red = cm[0, 1]  # factual & red
    good_red = cm[1, 1]  # hallucinating & red

    print("\nConfusion Matrix (Factual/Hallucinating vs. Green/Red):")
    print(f"  Good Green (Factual & Green)         : {good_green}")
    print(f"  Bad Green  (Hallucinating & Green)    : {bad_green}")
    print(f"  Bad Red    (Factual & Red)            : {bad_red}")
    print(f"  Good Red   (Hallucinating & Red)      : {good_red}")
