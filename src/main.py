import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any
import os

# Import your config flags/paths
from config import (
    DATASET_PATH,
    LOCAL_MODEL_NAME,
    DEFAULT_DATA_LIMIT,
    OUTPUT_DIR,
    ENABLE_DETAILED_LOGS,
)
from evaluator import evaluate_with_openai_api
from local_llm import (
    load_local_model,
    get_local_llm_answer,
    get_local_llm_hidden_states,
)
from hallucination_classifier import SimpleLinearClassifier


def load_dataset(dataset_path: str, data_limit: int = None) -> List[Dict[str, Any]]:
    """
    Loads the dataset from a JSON file and applies a data limit if specified.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if data_limit:
        data = data[:data_limit]
    return data


class LLMHiddenStateDataset(Dataset):
    """
    PyTorch Dataset that provides LLM hidden states and labels.
    Computes data on-the-fly to manage memory usage and avoid device issues.
    """

    def __init__(self, samples, model, tokenizer, layer_index=20):
        self.samples = samples
        self.model = model
        self.tokenizer = tokenizer
        self.layer_index = layer_index
        self.result_data = []  # Stores detailed results for logging/output

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item["question"]
        correct_answers = item["answers"]

        # Get local LLM answer
        local_answer = get_local_llm_answer(question, self.model, self.tokenizer)

        # Check if LLM answer is factual by simple string match
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)

        if matched:
            is_factual = True
            explanation = "string match"
        else:
            # Use OpenAI API to evaluate
            try:
                result = evaluate_with_openai_api(
                    question=question,
                    local_llm_answer=local_answer,
                    correct_answers=correct_answers,
                )
                is_factual = result["is_factual"]
                explanation = result.get("explanation", "")
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                is_factual = False
                explanation = f"API error: {e}"

        # Prepare the result entry
        result_entry = {
            "question": question,
            "answers": correct_answers,
            "llama_answer": local_answer,
            "is_factual": is_factual,
            "explanation": explanation,
        }

        # Save the result entry
        self.result_data.append(result_entry)
        if ENABLE_DETAILED_LOGS:
            print(
                f"[LOG] Detailed result entry:\n{json.dumps(result_entry, indent=2)}\n"
            )

        # Get hidden state from LLM
        hidden_state = get_local_llm_hidden_states(
            question, self.tokenizer, self.model, layer_index=self.layer_index
        )
        # Take last token's hidden state and flatten
        hidden_vector = hidden_state[:, -1, :].squeeze(0)  # shape (hidden_dim,)

        # Create label tensor (0 -> factual, 1 -> hallucinating)
        label = torch.tensor(0 if is_factual else 1, dtype=torch.long)

        return hidden_vector, label


def train_classifier(train_loader, input_dim, num_epochs=3, learning_rate=1e-4):
    """
    Trains the hallucination classifier using the provided training data.
    """
    classifier = SimpleLinearClassifier(input_dim=input_dim, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return classifier


def evaluate_classifier(classifier, test_loader):
    """
    Evaluates the classifier on the test set and prints out the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()
    total = 0
    correct = 0
    results = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device).float()
            labels = labels.to(device)

            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for idx in range(labels.size(0)):
                results.append(
                    {
                        "llm_is_factual": (labels[idx].item() == 0),
                        "classifier_pred_is_factual": (predicted[idx].item() == 0),
                    }
                )

    accuracy = 100 * correct / total
    print("\n\n ----------------------- \n\n")
    print(f"Classifier Accuracy on Test Set: {accuracy:.2f}%")

    # Summarize in four categories
    categories = {"Good Green": 0, "Bad Green": 0, "Bad Red": 0, "Good Red": 0}
    for res in results:
        llm_fact = res["llm_is_factual"]
        pred_fact = res["classifier_pred_is_factual"]
        if llm_fact and pred_fact:
            categories["Good Green"] += 1
        elif not llm_fact and pred_fact:
            categories["Bad Green"] += 1
        elif llm_fact and not pred_fact:
            categories["Bad Red"] += 1
        else:
            categories["Good Red"] += 1

    print("Evaluation Results:")
    for category, count in categories.items():
        print(f"{category}: {count}")

    return results


if __name__ == "__main__":
    # Seed everything for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_limit = DEFAULT_DATA_LIMIT

    # If user passed a command-line arg for data_limit:
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

    # Load the dataset
    dataset = load_dataset(DATASET_PATH, data_limit)

    # Load local LLM
    print("Loading local model...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # Create dataset
    data = LLMHiddenStateDataset(
        samples=dataset,
        model=model,
        tokenizer=tokenizer,
        layer_index=20,
    )

    # Process all data to fill result_data (and label/hidden states)
    print("Processing all data to collect results...")
    for idx in range(len(data)):
        _ = data[idx]  # trigger __getitem__
    print("Data processing complete.")

    # Train/test split (80/20)
    total_size = len(data)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        data, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # Dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Determine input dimension from a sample
    sample_feature, _ = data[0]
    input_dim = sample_feature.shape[0]

    # Train the classifier
    num_epochs = 3
    learning_rate = 1e-4
    classifier = train_classifier(train_loader, input_dim, num_epochs, learning_rate)

    # Evaluate on test set
    results = evaluate_classifier(classifier, test_loader)

    # Save output data
    output_file_path = os.path.join(OUTPUT_DIR, "output_data.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # data.result_data holds all Q/A info
    all_result_data = data.result_data
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(all_result_data, f, ensure_ascii=False, indent=4)

    print(f"Detailed results saved to '{output_file_path}'.")

# # colab commands:
# !unzip -q src.zip -d HiddenOracle3

# %cd HiddenOracle3/src

# !python main.py

# !python main.py 500
