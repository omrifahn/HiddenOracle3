import os
import json
import datetime
from typing import List, Dict, Any

import time

from openai import OpenAI # new sytax to import open api versions 1.0.0+
from pydantic import BaseModel

from local_llm import load_local_model, get_local_llm_answer
from config import OUTPUT_DIR, OPENAI_API_KEY, DATA_LIMIT

client = OpenAI(api_key=OPENAI_API_KEY) # new sytax for open api versions 1.0.0+


# 1. Load dataset
def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from a JSON file.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# 2. Evaluation functions
def evaluate_with_openai_api(
    question: str, local_llm_answer: str, correct_answers: List[str]
) -> Dict[str, Any]:
    """
    Calls the OpenAI ChatCompletion with Structured Outputs using JSON Schema
    to evaluate the local LLM's response.
    """

    class EvaluationResult(BaseModel):
        is_factual: bool
        explanation: str

    # Define the JSON schema based on the Pydantic model
    json_schema = {
        "type": "object",
        "properties": {
            "is_factual": {
                "type": "boolean",
                "description": "Indicates whether the local LLM's answer is factual.",
            },
            "explanation": {
                "type": "string",
                "description": "Explanation detailing the evaluation.",
            },
        },
        "required": ["is_factual", "explanation"],
        "additionalProperties": False,
    }

    # Convert JSON schema to a formatted string for inclusion in the prompt
    json_schema_str = json.dumps(json_schema, indent=4)

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": (
                "As an expert evaluator, compare the local LLM answer with the known correct answers. "
                "Provide your evaluation ensuring the output matches the specified JSON schema below.\n"
                f"JSON Schema:\n{json_schema_str}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Local LLM Answer: {local_llm_answer}\n"
                f"Known Correct Answers: {correct_answers}\n"
                "Provide your evaluation according to the schema."
            ),
        },
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",  # Use 'gpt-4' for better adherence to instructions
        messages=messages,
        temperature=0,
        max_tokens=500,
    )

    assistant_message = response.choices[0].message.content.strip()

    # Parse the assistant's response as JSON
    try:
        parsed_result = json.loads(assistant_message)
        # Validate the parsed result against the schema using Pydantic
        evaluation_result = EvaluationResult.parse_obj(parsed_result)
        return evaluation_result.dict()
    except json.JSONDecodeError as e:
        return {
            "is_factual": False,
            "explanation": f"Failed to parse response as JSON: {e}\nResponse: {assistant_message}",
        }


def evaluate_model(dataset_path: str, local_model_name: str) -> None:
    """
    1. Loads the dataset.
    2. Initializes the local model.
    3. Generates an answer with the local LLM.
    4. Uses either:
       - Simple string matching to determine correctness, or
       - OpenAI's evaluation with JSON Schema for deeper evaluation (if no match found).
    5. Saves the results to an output file.
    """
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Apply data limit
    if DATA_LIMIT is not None:
        dataset_size = len(dataset)
        if dataset_size < DATA_LIMIT:
            print(
                f"Dataset contains only {dataset_size} items. Processing all available items."
            )
            data_limit = dataset_size
        else:
            print(f"Processing first {DATA_LIMIT} items from the dataset.")
            data_limit = DATA_LIMIT
        dataset = dataset[:data_limit]
    else:
        data_limit = len(dataset)
        print(f"Processing all {data_limit} items from the dataset.")

    # Initialize local model pipeline
    generation_pipeline, model, tokenizer = load_local_model(local_model_name)

    # Prepare results list
    results = []

    # Evaluate each dataset item
    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        correct_answers = item["answers"]

        # 1) Generate answer from local LLM
        local_start_time = time.time()  # Start timing local LLM
        local_answer = get_local_llm_answer(question, generation_pipeline)
        local_end_time = time.time()  # End timing local LLM
        local_llm_runtime = local_end_time - local_start_time  # Calculate runtime

        # 2) First do a simple string match check to see if the local LLM's answer
        #    contains at least one of the known correct answers
        #    (case-insensitive substring check).
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)

        # 3) If we have a match, skip calling the OpenAI API and record a successful result.
        if matched:
            result = {
                "is_factual": True,
                "explanation": "String match with known correct answers; no OpenAI call needed.",
            }
            openai_api_runtime = 0  # No OpenAI API call made
        else:
            # 4) If no match, call the OpenAI API with JSON Schema for deeper evaluation.
            openai_start_time = time.time()  # Start timing OpenAI API call
            result = evaluate_with_openai_api(
                question=question,
                local_llm_answer=local_answer,
                correct_answers=correct_answers,
            )
            openai_end_time = time.time()  # End timing OpenAI API call
            openai_api_runtime = (
                openai_end_time - openai_start_time
            )  # Calculate runtime

        # 5) Prepare result entry
        result_entry = {
            "question": question,
            "correct_answers": correct_answers,
            "local_llm_answer": local_answer,
            "is_factual": result["is_factual"],
            "explanation": result["explanation"],
            "local_llm_runtime": local_llm_runtime,
            "openai_api_runtime": openai_api_runtime,
        }

        # Append to results
        results.append(result_entry)

        # Optionally, print results
        print("--------------------------------------------------")
        print(f"Question: {question}")
        print(f"Local LLM Answer: {local_answer}")
        print(f"Is Factual: {result['is_factual']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Local LLM Runtime: {local_llm_runtime:.4f} seconds")
        print(f"OpenAI API Runtime: {openai_api_runtime:.4f} seconds")
        print("--------------------------------------------------")

    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"run_results_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_path}")


# ---------------------------------------------------------
# __main__ sanity check
# ---------------------------------------------------------
if __name__ == "__main__":
    # Run evaluation
    print("Running sanity check for evaluator...")

    # For sanity check, create a sample dataset with mock local LLM answers
    sample_dataset = [
        {
            "question": "What is the capital of France?",
            "answers": ["Paris"],
            "local_llm_answer": "Paris",  # This should match via string match
        },
        {
            "question": "Who wrote '1984'?",
            "answers": ["George Orwell"],
            "local_llm_answer": "It was written in 1949.",  # Incorrect answer, requires OpenAI evaluation
        },
        {
            "question": "What is the tallest mountain in the world?",
            "answers": ["Mount Everest"],
            "local_llm_answer": "The tallest mountain is K2.",  # Incorrect, requires OpenAI evaluation
        },
    ]

    # Prepare results list
    results = []

    # Evaluate each sample item
    for item in sample_dataset:
        question = item["question"]
        correct_answers = item["answers"]
        local_answer = item["local_llm_answer"]

        # Check for string match
        matched = any(ans.lower() in local_answer.lower() for ans in correct_answers)

        if matched:
            result = {
                "is_factual": True,
                "explanation": "String match with known correct answers; no OpenAI call needed.",
            }
            openai_api_runtime = 0  # No OpenAI API call made
        else:
            # Call OpenAI API to evaluate
            openai_start_time = time.time()  # Start timing OpenAI API call
            result = evaluate_with_openai_api(
                question=question,
                local_llm_answer=local_answer,
                correct_answers=correct_answers,
            )
            openai_end_time = time.time()  # End timing OpenAI API call
            openai_api_runtime = (
                openai_end_time - openai_start_time
            )  # Calculate runtime

        # Prepare result entry
        result_entry = {
            "question": question,
            "correct_answers": correct_answers,
            "local_llm_answer": local_answer,
            "is_factual": result["is_factual"],
            "explanation": result["explanation"],
            "openai_api_runtime": openai_api_runtime,
        }

        # Append to results
        results.append(result_entry)

        # Print results
        print("--------------------------------------------------")
        print(f"Question: {question}")
        print(f"Local LLM Answer: {local_answer}")
        print(f"Is Factual: {result['is_factual']}")
        print(f"Explanation: {result['explanation']}")
        if openai_api_runtime > 0:
            print(f"OpenAI API Runtime: {openai_api_runtime:.4f} seconds")
        print("--------------------------------------------------")

    # Note: Removed the code that saves results to an output file
    print("Sanity check completed.")
