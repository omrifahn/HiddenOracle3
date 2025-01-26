"""
hallucination_classifier.py

Trains a simple linear classifier on top of the hidden layer 20 output
of the Llama model, in order to detect potential hallucinations.

Definitions:
-----------
- Good Green  : (LLM is factual,        classifier says factual)
- Bad Green   : (LLM is hallucinating,  classifier says factual)
- Bad Red     : (LLM is factual,        classifier says hallucinating)
- Good Red    : (LLM is hallucinating,  classifier says hallucinating)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import json
import os

from typing import List, Dict, Any
from local_llm import (
    load_local_model,
    get_local_llm_answer,
    get_local_llm_hidden_states
)
from config import (
    DATASET_PATH,
    DATA_LIMIT,
    LOCAL_MODEL_NAME,
    OUTPUT_DIR
)

# Optional: You may also wish to import the evaluation logic to decide if an LLM answer is factual:
# from evaluator import evaluate_with_openai_api

class HallucinationDataset(Dataset):
    """
    A PyTorch Dataset that:
      1) Loads question-answer pairs from a dataset JSON.
      2) Uses the local LLM to get the hidden states from layer 20.
      3) Assigns a binary label: 0 if "factual", 1 if "hallucinated."

    The labeling approach here is simplistic:
      - If local LLM answer is a direct string match with known correct answers (case-insensitive), label=0.
      - Otherwise, label=1.

    In a more advanced approach, you'd use an external evaluator (like GPT-4) or
    specialized logic to refine the labeling.
    """
    def __init__(
        self,
        model,
        tokenizer,
        dataset_path: str = DATASET_PATH,
        data_limit: int = DATA_LIMIT,
        layer_index: int = 20
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.layer_index = layer_index

        # Load dataset
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Apply data limit
        if data_limit is not None:
            data = data[:data_limit]

        self.samples = data

        # We'll store the hidden states and labels after computing them once
        self.hidden_state_cache = []
        self.label_cache = []

        # Precompute hidden states for all samples
        self._precompute()

    def _precompute(self):
        """
        Precomputes the hidden states and labels for all items in the dataset
        to avoid doing it repeatedly during training.
        """
        print("Precomputing hidden states for classifier dataset...")
        for idx, item in enumerate(self.samples):
            question = item["question"]
            correct_answers = item["answers"]

            # Get local LLM answer
            # We'll do a quick generation to see if it string-matches
            # any known correct answer. This is our "is factual" check.
            local_answer = get_local_llm_answer(
                question,
                None  # generation_pipeline is not needed if we do the pipeline step differently
            )
            # Because we changed load_local_model to return (pipeline, model, tokenizer),
            # you might want to build a separate pipeline or adapt your usage. For simplicity,
            # you could have a separate pipeline instance. We'll assume we pass it as None
            # for conceptual demonstration, but in actual code you'd pass the correct pipeline.

            # For minimal demonstration, let's say we do string matching:
            matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)
            # label=0 => factual, label=1 => hallucinated
            label = 0 if matched else 1

            # Now get hidden states from layer_index
            hidden_states = get_local_llm_hidden_states(
                question,
                self.tokenizer,
                self.model,
                self.layer_index
            )
            # Usually shape: (batch_size=1, seq_len, hidden_dim)
            # We'll reduce to a 1D vector for the linear classifier, e.g. by taking the last token
            # or an average pooling. Let's take the last token for simplicity:

            # hidden_states[:, -1, :] => shape: (1, hidden_dim)
            # We'll flatten it to shape: (hidden_dim,)
            feature_vector = hidden_states[:, -1, :].squeeze(0)  # shape (hidden_dim,)

            self.hidden_state_cache.append(feature_vector)
            self.label_cache.append(label)

        print("Precomputation finished.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a tuple: (feature_vector, label)
        feature_vector: shape (hidden_dim,)
        label: int in {0,1}
        """
        return self.hidden_state_cache[idx], self.label_cache[idx]


class SimpleLinearClassifier(nn.Module):
    """
    A simple linear classifier that predicts
    "0 => factual" or "1 => hallucinating."
    """
    def __init__(self, input_dim: int, num_labels: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        logits = self.linear(x)
        return logits


def train_hallucination_classifier(
    dataset_path: str = DATASET_PATH,
    data_limit: int = DATA_LIMIT,
    layer_index: int = 20,
    num_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 1e-4
):
    """
    Trains the linear classifier on top of Llama's hidden layer 20.
    Saves the trained classifier to disk for later usage.
    """
    # 1) Load local model (includes pipeline, model, tokenizer)
    print("Loading local model for classifier training...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Model loaded. Building dataset...")

    # 2) Build the PyTorch dataset
    dataset = HallucinationDataset(
        model=model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        data_limit=data_limit,
        layer_index=layer_index
    )

    # 3) Construct a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4) Instantiate a linear classifier
    # We have to guess input_dim from one sample. Let's get one item:
    sample_feature, _ = dataset[0]
    input_dim = sample_feature.shape[0]
    classifier = SimpleLinearClassifier(input_dim=input_dim, num_labels=2)
    classifier.train()

    # 5) Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # 6) Training loop
    print("Starting classifier training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for features, labels in dataloader:
            # features: (batch_size, input_dim)
            # labels: (batch_size)
            # Move to GPU if available
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
                classifier.cuda()

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

    # 7) Save the trained classifier
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classifier_path = os.path.join(OUTPUT_DIR, "hallucination_classifier.pt")
    torch.save(classifier.state_dict(), classifier_path)
    print(f"Trained classifier saved to {classifier_path}")


if __name__ == "__main__":
    """
    Example usage:
    This will train for a few epochs, storing the linear classifier in OUTPUT_DIR.
    """
    train_hallucination_classifier(
        dataset_path=DATASET_PATH,
        data_limit=DATA_LIMIT,
        layer_index=20,
        num_epochs=3,
        batch_size=8,
        learning_rate=1e-4
    )
