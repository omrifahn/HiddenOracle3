import sys
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


def main():
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    print("Loading enriched dataset from:", DATASET_PATH)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        enriched_data = json.load(f)

    features = []
    labels = []
    for item in enriched_data:
        features.append(item["hidden_vector"])
        labels.append(item["label"])
    features_np = np.array(features)
    labels_np = np.array(labels)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features_np,
        labels_np,
        test_size=1 - TRAIN_TEST_SPLIT_RATIO,
        random_state=seed,
        shuffle=True,
        stratify=labels_np,
    )

    train_factual_count = int((train_labels == 0).sum())
    train_halluc_count = int((train_labels == 1).sum())
    test_factual_count = int((test_labels == 0).sum())
    test_halluc_count = int((test_labels == 1).sum())

    print("\nData Balance:")
    print(f"  TRAIN set total: {len(train_labels)}")
    print(f"    Factual (0): {train_factual_count}")
    print(f"    Hallucinating (1): {train_halluc_count}")
    print(f"  TEST set total: {len(test_labels)}")
    print(f"    Factual (0): {test_factual_count}")
    print(f"    Hallucinating (1): {test_halluc_count}")

    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\nLogistic Regression Accuracy on Test Set: {accuracy * 100:.2f}%")

    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    good_green = int(cm[0, 0])
    bad_green = int(cm[1, 0])
    bad_red = int(cm[0, 1])
    good_red = int(cm[1, 1])
    print("\nConfusion Matrix (Factual/Hallucinating vs. Green/Red):")
    print(f"  Good Green (Factual & Green): {good_green}")
    print(f"  Bad Green (Hallucinating & Green): {bad_green}")
    print(f"  Bad Red (Factual & Red): {bad_red}")
    print(f"  Good Red (Hallucinating & Red): {good_red}")

    # Calculate additional metrics:
    precision_macro = precision_score(test_labels, predictions, average="macro")
    recall_macro = recall_score(test_labels, predictions, average="macro")
    f1_macro = f1_score(test_labels, predictions, average="macro")

    precision_weighted = precision_score(test_labels, predictions, average="weighted")
    recall_weighted = recall_score(test_labels, predictions, average="weighted")
    f1_weighted = f1_score(test_labels, predictions, average="weighted")

    # Get a detailed classification report:
    class_report = classification_report(test_labels, predictions, output_dict=True)

    print("\nAdditional Metrics:")
    print(f"  Precision (Macro): {precision_macro:.2f}")
    print(f"  Recall (Macro): {recall_macro:.2f}")
    print(f"  F1 Score (Macro): {f1_macro:.2f}")
    print(f"  Precision (Weighted): {precision_weighted:.2f}")
    print(f"  Recall (Weighted): {recall_weighted:.2f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.2f}")

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    # Build run report with a single "metrics" section for all evaluation metrics
    run_report = {
        "parameters": {
            "TRAIN_TEST_SPLIT_RATIO": TRAIN_TEST_SPLIT_RATIO,
        },
        "data_balance": {
            "train_total": len(train_labels),
            "train_factual_count": train_factual_count,
            "train_halluc_count": train_halluc_count,
            "test_total": len(test_labels),
            "test_factual_count": test_factual_count,
            "test_halluc_count": test_halluc_count,
        },
        "metrics": {
            "accuracy": accuracy,
            "confusion_matrix": {
                "good_green": good_green,
                "bad_green": bad_green,
                "bad_red": bad_red,
                "good_red": good_red,
            },
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "classification_report": class_report,
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_file_path = os.path.join(OUTPUT_DIR, "run_report_part_3.json")
    with open(report_file_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=4)
    print(f"\nRun report saved to '{report_file_path}'.")


if __name__ == "__main__":
    main()
