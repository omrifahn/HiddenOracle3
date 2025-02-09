#!/usr/bin/env python3
import json
import random
import os

# Adjust these file paths as needed.
# Here we assume the enriched data produced in Part 2 is saved as "hidden_state_data.json"
# (make sure to update the path if needed).
INPUT_FILE = os.path.join(os.path.dirname(__file__), "hidden_state_data.json")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "balanced_data.json")


def load_hidden_state_data(file_path):
    """Load the hidden state data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def balance_data(data):
    """
    Balance the dataset by undersampling the majority class.

    It assumes that each item has a 'label' key where:
      - 0 indicates a factual (green) sample,
      - 1 indicates a hallucinating (red) sample.
    """
    # Separate samples by class
    factual_samples = [item for item in data if item.get("label") == 0]
    halluc_samples = [item for item in data if item.get("label") == 1]

    num_factual = len(factual_samples)
    num_halluc = len(halluc_samples)
    print(f"Original counts: factual = {num_factual}, hallucinating = {num_halluc}")

    # Determine the count of the minority class
    minority_count = min(num_factual, num_halluc)

    # Undersample the majority class
    if num_factual > num_halluc:
        factual_samples = random.sample(factual_samples, minority_count)
    elif num_halluc > num_factual:
        halluc_samples = random.sample(halluc_samples, minority_count)

    # Combine and shuffle the balanced samples
    balanced = factual_samples + halluc_samples
    random.shuffle(balanced)
    print(
        f"Balanced counts: factual = {sum(1 for item in balanced if item.get('label') == 0)}, "
        f"hallucinating = {sum(1 for item in balanced if item.get('label') == 1)}"
    )
    return balanced


def save_balanced_data(data, file_path):
    """Save the balanced dataset to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Balanced data saved to '{file_path}'.")


def main():
    # Load the data from the enriched JSON file (Part 2 output)
    data = load_hidden_state_data(INPUT_FILE)

    # Balance the dataset by undersampling the majority class
    balanced = balance_data(data)

    # Save the balanced dataset to a new JSON file
    save_balanced_data(balanced, OUTPUT_FILE)


if __name__ == "__main__":
    main()
