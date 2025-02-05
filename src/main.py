import sys
from config import DATASET_PATH, LOCAL_MODEL_NAME, DEFAULT_DATA_LIMIT
from evaluator import evaluate_with_openai_api
from local_llm import (
    load_local_model,
    get_local_llm_answer,
    get_local_llm_hidden_states,
)
from hallucination_classifier import SimpleLinearClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import os
import json
import time
from typing import List, Dict, Any
import random
import numpy as np


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

    def __init__(self, samples, model, tokenizer, generation_pipeline, layer_index=20):
        self.samples = samples
        self.model = model
        self.tokenizer = tokenizer
        self.generation_pipeline = generation_pipeline
        self.layer_index = layer_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        question = item["question"]
        correct_answers = item["answers"]

        # Get local LLM answer
        local_answer = get_local_llm_answer(question, self.generation_pipeline)

        # Determine if LLM's answer is factual
        # First, try simple string match
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)

        if matched:
            is_factual = True
        else:
            # Use OpenAI API to evaluate
            try:
                result = evaluate_with_openai_api(
                    question=question,
                    local_llm_answer=local_answer,
                    correct_answers=correct_answers,
                )
                is_factual = result["is_factual"]
            except Exception as e:
                print(f"Error during OpenAI API call: {e}")
                is_factual = False  # Default to not factual if API call fails

        # Get hidden state from LLM
        hidden_state = get_local_llm_hidden_states(
            question, self.tokenizer, self.model, layer_index=self.layer_index
        )
        # Take last token's hidden state and flatten
        hidden_vector = hidden_state[:, -1, :].squeeze(0)  # shape (hidden_dim,)

        # Create label tensor
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
            features = features.to(device).float()  # Convert features to float32
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
            features = features.to(device).float()  # Convert features to float32
            labels = labels.to(device)

            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for idx in range(labels.size(0)):
                results.append(
                    {
                        "llm_is_factual": labels[idx].item()
                        == 0,  # True if label is 0 (factual)
                        "classifier_pred_is_factual": predicted[idx].item()
                        == 0,  # True if pred is 0
                    }
                )

    accuracy = 100 * correct / total
    print(f"Classifier Accuracy on Test Set: {accuracy:.2f}%")

    # Count occurrences in four categories
    categories = {"Good Green": 0, "Bad Green": 0, "Bad Red": 0, "Good Red": 0}

    for res in results:
        llm_is_factual = res["llm_is_factual"]
        classifier_pred_is_factual = res["classifier_pred_is_factual"]

        if llm_is_factual and classifier_pred_is_factual:
            categories["Good Green"] += 1
        elif not llm_is_factual and classifier_pred_is_factual:
            categories["Bad Green"] += 1
        elif llm_is_factual and not classifier_pred_is_factual:
            categories["Bad Red"] += 1
        elif not llm_is_factual and not classifier_pred_is_factual:
            categories["Good Red"] += 1

    print("Evaluation Results:")
    for category, count in categories.items():
        print(f"{category}: {count}")

    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_limit = DEFAULT_DATA_LIMIT

    # Override data_limit if provided as command-line argument
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if arg1.lower() == "none":
            data_limit = None
        else:
            try:
                data_limit = int(arg1)
            except ValueError:
                print(
                    f"Invalid data_limit '{arg1}' provided. Using default value {data_limit}."
                )

    # Load dataset with specified data limit
    dataset = load_dataset(DATASET_PATH, data_limit)

    # Load local LLM model
    print("Loading local model...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # Prepare dataset
    data = LLMHiddenStateDataset(
        samples=dataset,
        model=model,
        tokenizer=tokenizer,
        generation_pipeline=generation_pipeline,
        layer_index=20,
    )

    # Split into training and testing datasets (80% train, 20% test)
    total_size = len(data)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        data, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get input dimension from one sample
    sample_feature, _ = data[0]
    input_dim = sample_feature.shape[0]

    # Train classifier
    num_epochs = 3
    learning_rate = 1e-4
    classifier = train_classifier(train_loader, input_dim, num_epochs, learning_rate)

    # Evaluate classifier
    results = evaluate_classifier(classifier, test_loader)


# # colab commands:
# !unzip -q src.zip -d HiddenOracle3

# %cd HiddenOracle3/src

# !python main.py

# !python main.py 500
