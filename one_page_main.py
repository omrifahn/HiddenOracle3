# -*- coding: utf-8 -*-
"""
HiddenOracle_StructuredOutputs.ipynb

Evaluates a local LLM's factual/hallucination answers,
trains a linear classifier on hidden states, logs results,
and uses OpenAI's Structured Outputs via JSON format.

"""

# =========================
# Basic Setup & Configuration
# =========================

import os
import openai
from datetime import datetime

# Ensure your OpenAI library is up to date
# !pip install --upgrade openai

# Set your Hugging Face token (for Llama-2) in the environment variable
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "your-huggingface-token"

# Set your OpenAI API key in the environment variable
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Get OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'")
openai.api_key = OPENAI_API_KEY

# Create a date/time stamp for unique output on each run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# We'll store the final JSON results in an "output" subfolder:
OUTPUT_JSON_PATH = f"./output/evaluation_results_{timestamp}.json"

# Example dataset with ~500 examples:
DATASET_PATH = "./Data/500_examples_data.json"

# The local Llama-2 model:
LOCAL_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Cache settings:
MODEL_CACHE_DIR = "./models"
USE_LOCAL_CACHE = True
FORCE_REDOWNLOAD = False

# Maximum number of samples to process:
MAX_SAMPLES = 100  # Set to None or a larger number to use more data

print(f"Timestamp: {timestamp}")
print(f"Output JSON: {OUTPUT_JSON_PATH}")

# =========================
# Main Logic
# =========================

import random
import json
import time
import torch
import numpy as np
import openai
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

########################################
# A) Load Dataset (with optional sampling)
########################################

def load_dataset(dataset_path: str, max_samples: int = None):
    """
    Loads the dataset from a JSON file.
    If max_samples is set, randomly sample that many items.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None and max_samples < len(data):
        data = random.sample(data, k=max_samples)

    return data

########################################
# B) Local LLM Generation
########################################

def get_local_llm_answer(question: str, generation_pipeline) -> str:
    """
    Use the local Llama model pipeline to generate an answer.
    We'll do 25 new tokens, do_sample=False for a short deterministic answer.
    """
    results = generation_pipeline(
        question,
        max_new_tokens=25,
        num_return_sequences=1,
        do_sample=False
    )
    generated_text = results[0]["generated_text"]

    # If LLM repeated the prompt at the start, remove it
    if generated_text.startswith(question):
        return generated_text[len(question):].strip()
    else:
        return generated_text.strip()

########################################
# C) OpenAI Evaluation (Structured Outputs via JSON Format)
########################################

def evaluate_with_openai_api(question: str, local_llm_answer: str, correct_answers: list) -> dict:
    """
    Use OpenAI to evaluate the correctness of the local LLM answer.
    The model's response will follow a specified JSON format:
    {
        "reasoning": "<string>",
        "Judgement": <true/false>
    }
    """
    # Define the messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a judge who compares the local LLM's answer with known correct answers.\n"
                "Provide your reasoning and judgement in the specified JSON format.\n"
                "Format the response as a JSON object with the following keys:\n"
                "- reasoning: string\n"
                "- Judgement: boolean (true if the local LLM's answer is correct, false if incorrect)\n"
                "Ensure that the response is a valid JSON object and contains all required fields.\n"
                "Do not include any additional text."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Local LLM Answer: {local_llm_answer}\n"
                f"Known Correct Answers: {correct_answers}"
            )
        }
    ]

    try:
        # Call the OpenAI API
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
        )

        # Get the assistant's message content
        response_content = completion.choices[0].message['content']

        # Parse the JSON content
        parsed_response = json.loads(response_content)
        return {
            "reasoning": parsed_response.get("reasoning", ""),
            "Judgement": parsed_response.get("Judgement", False)
        }
    except Exception as e:
        # Handle any exceptions, such as JSON parsing errors
        return {
            "reasoning": f"Error during evaluation: {str(e)}",
            "Judgement": False
        }

########################################
# D) Load Local Model with Hidden States
########################################

def load_local_llm_with_hidden(model_name: str):
    """
    Loads Llama-2 from HF, with output_hidden_states=True, device_map='auto'.
    """
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise ValueError("No Hugging Face token found (HUGGINGFACE_HUB_TOKEN).")

    from_pretrained_kwargs = {}
    if USE_LOCAL_CACHE:
        from_pretrained_kwargs["cache_dir"] = MODEL_CACHE_DIR
    if FORCE_REDOWNLOAD:
        from_pretrained_kwargs["force_download"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        **from_pretrained_kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        output_hidden_states=True,
        **from_pretrained_kwargs
    )

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        temperature=0.6,  # This won't matter if do_sample=False
        top_p=0.9,
        device=model.device
    )

    return model, tokenizer, generation_pipeline

########################################
# E) Get Hidden State from Layer 20
########################################

def get_hidden_state_layer_20(model, tokenizer, text: str):
    """
    Tokenize 'text' and get the last token's representation from layer 20.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.hidden_states[20] => shape: [1, seq_len, hidden_dim]
        layer_20_rep = outputs.hidden_states[20]
    return layer_20_rep[0, -1, :]  # Last token

########################################
# F) Main Evaluate + Classifier + Confusion Matrix
########################################

def evaluate_model(dataset_path: str, local_model_name: str, output_json_path: str, max_samples: int):
    """
    Steps:
    1) Load dataset
    2) Load local LLM
    3) For each question:
       - Generate local answer
       - If substring match => factual
         else => call GPT-4 to decide using Structured Outputs
       - Get layer-20 hidden state
    4) Train logistic regression
    5) Compute confusion matrix
    6) Save results to JSON
    """
    start_run = time.time()

    # 1) Load data
    dataset = load_dataset(dataset_path, max_samples)

    # 2) Load local model/pipeline
    model, tokenizer, pipeline_llm = load_local_llm_with_hidden(local_model_name)

    evaluation_results = []
    X = []
    y = []  # 0 => factual, 1 => hallucination

    for idx, item in enumerate(dataset):
        question = item["question"]
        correct_answers = item["answers"]

        # Local generation
        t0_local = time.time()
        local_answer = get_local_llm_answer(question, pipeline_llm)
        local_time = time.time() - t0_local

        # Check substring
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)

        # If matched => factual, else => GPT evaluation
        t0_openai = time.time()
        if matched:
            result = {
                "reasoning": "Answer matches known correct answers.",
                "Judgement": True
            }
            openai_time = 0.0
        else:
            result = evaluate_with_openai_api(question, local_answer, correct_answers)
            openai_time = time.time() - t0_openai

        label = 0 if result["Judgement"] else 1  # 0 for factual, 1 for hallucination
        item_time = local_time + openai_time

        print(f"Item {idx+1}/{len(dataset)}")
        print("--------------------------------------------------")
        print(f"Q: {question}")
        print(f"Local LLM Answer: {local_answer}")
        print(f"Judgement: {result['Judgement']} => Reasoning: {result['reasoning']}")
        print(f"Local LLM Time (s):  {local_time:.4f}")
        print(f"OpenAI Time (s):     {openai_time:.4f}")
        print(f"Item Total (s):      {item_time:.4f}")
        print("--------------------------------------------------")

        evaluation_results.append({
            "question": question,
            "answers": correct_answers,
            "local_answer": local_answer,
            "Judgement": result["Judgement"],
            "reasoning": result["reasoning"],
            "local_llm_time_seconds": round(local_time, 4),
            "openai_time_seconds": round(openai_time, 4),
            "item_total_time_seconds": round(item_time, 4)
        })

        # Get hidden state layer 20
        hidden_vec = get_hidden_state_layer_20(model, tokenizer, question)
        hidden_np = hidden_vec.cpu().numpy()

        X.append(hidden_np)
        y.append(label)

    # 4) Train logistic regression
    X_array = np.array(X)
    y_array = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    # 5) Confusion matrix
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Interpret the 2x2 matrix
    print("=== Confusion Matrix ===")
    print("Labels: [Factual=0, Hallucination=1]")
    print(cm)
    print(f"TN (True Negative - Correctly identified factual): {tn}")
    print(f"FP (False Positive - Incorrectly labeled hallucination as factual):    {fp}")
    print(f"FN (False Negative - Incorrectly labeled factual as hallucination):  {fn}")
    print(f"TP (True Positive - Correctly identified hallucination):   {tp}\n")
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")

    # 6) Save results to JSON
    total_run = time.time() - start_run
    output_data = {
        "evaluation_results": evaluation_results,
        "classifier_performance": {
            "accuracy": accuracy,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        },
        "metadata": {
            "total_run_time_seconds": round(total_run, 4),
            "max_samples_used": max_samples,
            "dataset_path": dataset_path,
            "timestamp": timestamp
        }
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_json_path}")
    print(f"Total run time: {total_run:.4f} seconds")
    print("Done with evaluation + classifier training + confusion matrix.")

# =========================
# Run the evaluation
# =========================

if __name__ == "__main__":
    evaluate_model(
        dataset_path=DATASET_PATH,
        local_model_name=LOCAL_MODEL_NAME,
        output_json_path=OUTPUT_JSON_PATH,
        max_samples=MAX_SAMPLES
    )

print("Main code executed if __name__ == '__main__'.")
