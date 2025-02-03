import json
from typing import List, Dict, Any

from openai import OpenAI # new sytax to import open api versions 1.0.0+
from pydantic import BaseModel

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY) # new sytax for open api versions 1.0.0+


def evaluate_with_openai_api(
    question: str, local_llm_answer: str, correct_answers: List[str]
) -> Dict[str, Any]:
    """
    Calls the OpenAI API to evaluate the local LLM's response.
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
