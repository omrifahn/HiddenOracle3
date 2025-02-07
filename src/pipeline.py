# pipeline.py

import json
import torch
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
      2) Determines factuality (via simple string match or OpenAI evaluator)
      3) Extracts final hidden states from the LLM
      4) Stores features (hidden states), labels (0 or 1), and result logs

    Returns:
      all_features_np: np.ndarray [num_samples, hidden_dim]
      all_labels_np:   np.ndarray [num_samples]
      result_data:     List[dict] with question, answers, llama_answer, is_factual, explanation
    """
    all_features = []
    all_labels = []
    result_data = []

    for item in samples:
        question = item["question"]
        correct_answers = item["answers"]

        # 1) Generate LLM answer
        local_answer = get_local_llm_answer(question, model, tokenizer)

        # 2) Check if it's factual (simple match or fallback to OpenAI evaluator)
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

        # Build a result entry for logging
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

        # 3) Extract the final hidden states from the model
        hidden_state = get_local_llm_hidden_states(
            question, tokenizer, model, layer_index
        )
        # Take the last token's hidden state
        hidden_vector = hidden_state[:, -1, :].squeeze(0)  # shape: (hidden_dim,)

        # 4) 0 => factual, 1 => hallucinating
        label = 0 if is_factual else 1

        all_features.append(hidden_vector)
        all_labels.append(label)

    # Convert to PyTorch Tensors, then to NumPy for scikit-learn
    all_features_tensor = torch.stack(all_features).cpu().float()
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Convert to NumPy
    all_features_np = all_features_tensor.numpy()
    all_labels_np = all_labels_tensor.numpy()

    return all_features_np, all_labels_np, result_data
