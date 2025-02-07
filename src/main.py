import sys
import random
import numpy as np
import torch
import os
import json

from torch.utils.data import TensorDataset, DataLoader, random_split

from config import (
    DATASET_PATH,
    LOCAL_MODEL_NAME,
    DEFAULT_DATA_LIMIT,
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    LAYER_INDEX,
    TRAIN_TEST_SPLIT_RATIO,
)

from pipeline import (
    load_dataset,
    precompute_hidden_states_and_labels,
    train_classifier,
    evaluate_classifier,
)

from local_llm import (
    load_local_model,
)

from hallucination_classifier import SimpleLinearClassifier


if __name__ == "__main__":
    # Seed everything for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_limit = DEFAULT_DATA_LIMIT

    # If user passed a command-line arg for data_limit
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1.lower() == "none":
            data_limit = None
        else:
            try:
                data_limit = int(arg1)
            except ValueError:
                print(f"Invalid data_limit '{arg1}' provided. Using default {data_limit}.")

    # 1) Load raw dataset
    dataset = load_dataset(DATASET_PATH, data_limit)

    # 2) Load local LLM
    print("Loading local model...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # 3) Precompute hidden states + factual labels in a single pass
    print("Precomputing hidden states and labels...")
    features, labels, result_data = precompute_hidden_states_and_labels(
        samples=dataset, model=model, tokenizer=tokenizer, layer_index=LAYER_INDEX
    )
    print("Precomputation complete.")

    # (Optional) log or save these results
    output_file_path = os.path.join(OUTPUT_DIR, "output_data.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print(f"Detailed results saved to '{output_file_path}'.")

    # 4) Create a TensorDataset from precomputed features & labels
    full_dataset = TensorDataset(features, labels)

    # 5) Train/test split
    total_size = len(full_dataset)
    train_size = int(TRAIN_TEST_SPLIT_RATIO * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # 6) Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 7) Get input dimension from the first sample
    sample_feature, _ = train_dataset[0]
    input_dim = sample_feature.shape[0]

    # 8) Initialize and train the classifier
    classifier = SimpleLinearClassifier(input_dim=input_dim, num_labels=2)
    classifier = train_classifier(
        train_loader, classifier,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )

    # 9) Evaluate classifier
    _ = evaluate_classifier(classifier, test_loader)
