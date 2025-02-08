# File: hidden_state_extraction/evaluator.py
import json
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def evaluate_with_openai_api(
    question: str, local_llm_answer: str, correct_answers: List[str]
) -> Dict[str, Any]:
    class EvaluationResult(BaseModel):
        is_factual: bool
        explanation: str

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
    json_schema_str = json.dumps(json_schema, indent=4)
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
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=500,
    )
    assistant_message = response.choices[0].message.content.strip()
    try:
        parsed_result = json.loads(assistant_message)
        evaluation_result = EvaluationResult.parse_obj(parsed_result)
        return evaluation_result.dict()
    except json.JSONDecodeError as e:
        return {
            "is_factual": False,
            "explanation": f"Failed to parse response as JSON: {e}\nResponse: {assistant_message}",
        }
