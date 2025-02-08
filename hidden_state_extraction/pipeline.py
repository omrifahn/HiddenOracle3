import json
import torch
import time
from typing import List, Dict, Any
from .evaluator import evaluate_with_openai_api
from .local_llm import get_local_llm_answer, get_local_llm_hidden_states
from .config import ENABLE_DETAILED_LOGS


def load_dataset(dataset_path: str, data_limit: int = None) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if data_limit:
        data = data[:data_limit]
    return data


def precompute_hidden_states_and_labels(samples, model, tokenizer, layer_index=20):
    updated_data = []
    error_log = []
    # Initialize counters and list for runtimes.
    summary_stats = {
        "total_samples": len(samples),
        "factual_string_match": 0,
        "factual_api": 0,
        "hallucinations": 0,
        "runtime_per_datapoint": [],
    }

    for item in samples:
        start_time = time.time()
        question = item["question"]
        correct_answers = item["answers"]

        try:
            local_answer = get_local_llm_answer(question, model, tokenizer)
        except Exception as e:
            error_log.append(
                {"question": question, "error": f"Error generating LLM answer: {e}"}
            )
            continue

        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)
        if matched:
            is_factual = True
            explanation = "string match"
            evaluation_method = "string_match"
            summary_stats["factual_string_match"] += 1
        else:
            try:
                eval_result = evaluate_with_openai_api(
                    question, local_answer, correct_answers
                )
                is_factual = eval_result["is_factual"]
                explanation = eval_result.get("explanation", "")
                evaluation_method = "openai_api"
                if is_factual:
                    summary_stats["factual_api"] += 1
                else:
                    summary_stats["hallucinations"] += 1
            except Exception as e:
                is_factual = False
                explanation = f"Evaluation error: {e}"
                evaluation_method = "evaluation_error"
                summary_stats["hallucinations"] += 1

        try:
            with torch.no_grad():
                hidden_state = get_local_llm_hidden_states(
                    question, tokenizer, model, layer_index
                )
            hidden_vector = hidden_state[:, -1, :].squeeze(0).tolist()
        except Exception as e:
            error_log.append(
                {"question": question, "error": f"Error extracting hidden state: {e}"}
            )
            continue

        label = 0 if is_factual else 1
        runtime = time.time() - start_time
        summary_stats["runtime_per_datapoint"].append(runtime)

        item.update(
            {
                "llama_answer": local_answer,
                "is_factual": is_factual,
                "explanation": explanation,
                "hidden_vector": hidden_vector,
                "label": label,
                "processing_time": runtime,
                "evaluation_method": evaluation_method,
            }
        )

        if ENABLE_DETAILED_LOGS:
            log_item = {
                "question": item.get("question"),
                "llama_answer": item.get("llama_answer"),
                "is_factual": item.get("is_factual"),
                "explanation": item.get("explanation"),
                "evaluation_method": evaluation_method,
                "processing_time": runtime,
            }
            print(f"[LOG] Updated sample:\n{json.dumps(log_item, indent=2)}\n")

        updated_data.append(item)

    return updated_data, error_log, summary_stats
