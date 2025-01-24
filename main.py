import json
from typing import List, Dict, Any
from openai import OpenAI

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import config
from config import OPENAI_API_KEY, DATASET_PATH, LOCAL_MODEL_NAME, configure_openai


# ---------------------------------------------------------
# 1. Load dataset and local model generation
# ---------------------------------------------------------
def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from a JSON file.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def get_local_llm_answer(question: str, generation_pipeline) -> str:
    """
    Uses a text-generation pipeline to get an answer from the local LLM.
    """
    results = generation_pipeline(
        question, max_new_tokens=25, num_return_sequences=1, do_sample=False
    )
    generated_text = results[0]["generated_text"]

    # Optionally trim out the prompt if it's included in the output
    if generated_text.startswith(question):
        answer = generated_text[len(question) :].strip()
    else:
        answer = generated_text.strip()

    return answer


# ---------------------------------------------------------
# 2. Evaluate correctness with the NEW function calling (tools)
# ---------------------------------------------------------
def evaluate_with_openai_api(
    question: str, local_llm_answer: str, correct_answers: List[str]
) -> Dict[str, Any]:
    """
    Calls the OpenAI ChatCompletion with the new function-calling style to evaluate the local LLM's response.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "evaluate_answer",
                "description": "Evaluate the correctness of a local LLM's answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_factual": {
                            "type": "boolean",
                            "description": "True if the local LLM answer is correct.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Explanation of why the answer is correct or incorrect.",
                        },
                    },
                    "required": ["is_factual", "explanation"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a judge. Compare the local LLM answer with known correct answers. "
                "Use the evaluate_answer function to return factuality and reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Local LLM Answer: {local_llm_answer}\n"
                f"Known Correct Answers: {correct_answers}"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "evaluate_answer"}},
    )

    if completion.choices[0].message.tool_calls:
        tool_call = completion.choices[0].message.tool_calls[0]
        # Correctly access the tool_call details
        arguments = json.loads(
            tool_call.function.arguments
        )  # Assuming .function.arguments is correct
        return arguments
    else:
        return {"is_factual": False, "explanation": "No tool call was made."}


# ---------------------------------------------------------
# 3. Main evaluation routine
# ---------------------------------------------------------
def evaluate_model(dataset_path: str, local_model_name: str) -> None:
    """
    1. Loads the dataset.
    2. Initializes the local model.
    3. Generates an answer with the local LLM.
    4. Uses OpenAI's function calling to judge correctness.
    """
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Initialize local model pipeline
    tokenizer = AutoTokenizer.from_pretrained(local_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Evaluate each dataset item
    for item in dataset:
        question = item["question"]
        correct_answers = item["answers"]

        # 1) Generate answer from local LLM
        local_answer = get_local_llm_answer(question, generation_pipeline)
        # local_answer = "im not sure about that"

        # 2) Evaluate correctness via OpenAI function calling
        result = evaluate_with_openai_api(
            question=question,
            local_llm_answer=local_answer,
            correct_answers=correct_answers,
        )

        # 3) Print results
        print("--------------------------------------------------")
        print(f"Question: {question}")
        print(f"Local LLM Answer: {local_answer}")
        print(f"Is Factual: {result['is_factual']}")
        print(f"Explanation: {result['explanation']}")
        print("--------------------------------------------------")


# ---------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Configure OpenAI
    configure_openai()

    # Run evaluation
    evaluate_model(DATASET_PATH, LOCAL_MODEL_NAME)
