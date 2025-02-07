import sys
import random
import numpy as np
import torch
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from config import (
    DATASET_PATH,
    LOCAL_MODEL_NAME,
    DEFAULT_DATA_LIMIT,
    OUTPUT_DIR,
    LAYER_INDEX,
    TRAIN_TEST_SPLIT_RATIO,
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

    # 2) Load local LLM
    print("Loading local model...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # 3) Precompute hidden states + factual labels
    print("Precomputing hidden states and labels...")
    features, labels, result_data = precompute_hidden_states_and_labels(
        samples=dataset, model=model, tokenizer=tokenizer, layer_index=LAYER_INDEX
    )
    print("Precomputation complete.")

    # (Optional) save result_data
    output_file_path = os.path.join(OUTPUT_DIR, "output_data.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print(f"Detailed results saved to '{output_file_path}'.")

    # 4) Train/test split
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO,
        random_state=seed,
        shuffle=True,
    )

    # 5) Train logistic regression
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_labels)

    # 6) Evaluate
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nLogistic Regression Accuracy on Test Set: {accuracy * 100:.2f}%")

    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:\n", cm)
