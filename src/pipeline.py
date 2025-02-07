# pipeline.py

import json
import torch
import random
import numpy as np

from typing import List, Dict, Any
from evaluator import evaluate_with_openai_api
from local_llm import (
    get_local_llm_answer,
    get_local_llm_hidden_states,
)
from config import ENABLE_DETAILED_LOGS


def load_dataset(dataset_path: str, data_limit: int = None) -> List[Dict[str, Any]]:
    """
    Loads the dataset from a JSON file and applies a data limit if specified.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if data_limit:
        data = data[:data_limit]
    return data


def precompute_hidden_states_and_labels(samples, model, tokenizer, layer_index=20):
    """
    Single-pass function that:
      1) Generates local LLM answers for each sample
      2) Determines factuality (via simple match or OpenAI evaluator)
      3) Extracts final hidden states from the LLM
      4) Stores features (hidden states), labels (0 or 1), and result logs

    Returns:
      all_features: FloatTensor [num_samples, hidden_dim]
      all_labels: LongTensor [num_samples]
      result_data: List[dict] with question, answers, llama_answer, is_factual, explanation
    """
    all_features = []
    all_labels = []
    result_data = []

    for item in samples:
        question = item["question"]
        correct_answers = item["answers"]

        # 1) Generate LLM answer
        local_answer = get_local_llm_answer(question, model, tokenizer)

        # 2) Check if it's factual (simple match -> fallback to OpenAI evaluator)
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)
        if matched:
            is_factual = True
            explanation = "string match"
        else:
            try:
                eval_result = evaluate_with_openai_api(
                    question, local_answer, correct_answers
                )
                is_factual = eval_result["is_factual"]
                explanation = eval_result.get("explanation", "")
            except Exception as e:
                is_factual = False
                explanation = f"API error: {e}"

        # Save logs
        entry = {
            "question": question,
            "answers": correct_answers,
            "llama_answer": local_answer,
            "is_factual": is_factual,
            "explanation": explanation,
        }
        result_data.append(entry)
        if ENABLE_DETAILED_LOGS:
            print(f"[LOG] Detailed result entry:\n{json.dumps(entry, indent=2)}\n")

        # 3) Get the final hidden states
        hidden_state = get_local_llm_hidden_states(
            question, tokenizer, model, layer_index
        )
        hidden_vector = hidden_state[:, -1, :].squeeze(0)  # shape (hidden_dim,)

        # 4) Build label (0 => factual, 1 => hallucinating)
        label = 0 if is_factual else 1

        all_features.append(hidden_vector)
        all_labels.append(label)

    # Convert to PyTorch tensors
    all_features = torch.stack(all_features).cpu().float()
    all_labels = torch.tensor(all_labels, dtype=torch.long)

    return all_features, all_labels, result_data


def train_classifier(train_loader, classifier, num_epochs=3, learning_rate=1e-4):
    """
    Trains the hallucination classifier using the provided training data.
    """
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward(retain_graph=True)  # Avoid clearing the graph
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return classifier


def evaluate_classifier(classifier, test_loader):
    """
    Evaluates the classifier on the test set and prints out the results.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()
    total = 0
    correct = 0
    results = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                results.append(
                    {
                        "llm_is_factual": (labels[i].item() == 0),
                        "classifier_pred_is_factual": (predicted[i].item() == 0),
                    }
                )

    accuracy = 100.0 * correct / total
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

