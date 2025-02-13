import random
import numpy as np
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from config import DATASET_PATH, OUTPUT_DIR, TRAIN_TEST_SPLIT_RATIO


def train_classifier_for_layer(layer_index, features_np, labels_np):
    # Split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features_np,
        labels_np,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO,
        random_state=42,
        shuffle=True,
        stratify=labels_np,
    )
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    precision_macro = precision_score(test_labels, predictions, average="macro")
    recall_macro = recall_score(test_labels, predictions, average="macro")
    f1_macro = f1_score(test_labels, predictions, average="macro")
    precision_weighted = precision_score(test_labels, predictions, average="weighted")
    recall_weighted = recall_score(test_labels, predictions, average="weighted")
    f1_weighted = f1_score(test_labels, predictions, average="weighted")
    class_report = classification_report(test_labels, predictions, output_dict=True)

    result = {
        "layer": layer_index,
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_negatives": int(cm[1, 0]),
            "false_positives": int(cm[0, 1]),
            "true_positives": int(cm[1, 1]),
        },
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": class_report,
    }
    return result


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    print("Loading enriched dataset from:", DATASET_PATH)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    # Determine available layer indices from the first sample.
    # Note: When stored as JSON, dictionary keys become strings.
    sample_hidden_vectors = enriched_data[0]["hidden_vectors"]
    layer_indices = sorted([int(key) for key in sample_hidden_vectors.keys()])

    layer_results = []
    for layer in layer_indices:
        features = []
        labels = []
        for item in enriched_data:
            # Convert the layer index to a string because keys are strings in JSON.
            features.append(item["hidden_vectors"][str(layer)])
            labels.append(item["label"])
        features_np = np.array(features)
        labels_np = np.array(labels)
        result = train_classifier_for_layer(layer, features_np, labels_np)
        layer_results.append(result)
        print(f"Layer {layer}: Accuracy = {result['accuracy'] * 100:.2f}%")

    run_report = {"layer_results": layer_results}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_file_path = os.path.join(OUTPUT_DIR, "run_report_all_layers.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=4)
    print(f"\nRun report saved to '{report_file_path}'.")


if __name__ == "__main__":
    main()
