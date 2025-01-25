import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    HUGGINGFACE_TOKEN,
    LOCAL_MODEL_NAME,
    LOCAL_MODEL_DIR,
    USE_LOCAL_MODEL_STORAGE,
)


def load_local_model(local_model_name: str):
    """
    Loads the local LLM and returns a text generation pipeline.
    If USE_LOCAL_MODEL_STORAGE is True and the model is already downloaded in LOCAL_MODEL_DIR, it loads from storage.
    Otherwise, it downloads the model and saves it locally to save internet downloads in the future.
    """
    if USE_LOCAL_MODEL_STORAGE:
        # Replace slashes in model name to create a valid directory name
        model_dir_name = local_model_name.replace("/", "_")
        model_path = os.path.join(LOCAL_MODEL_DIR, model_dir_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(
                f"Model not found locally. Downloading '{local_model_name}' to '{model_path}'..."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_name, use_auth_token=HUGGINGFACE_TOKEN, cache_dir=model_path
            )
            model = AutoModelForCausalLM.from_pretrained(
                local_model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_auth_token=HUGGINGFACE_TOKEN,
                cache_dir=model_path,
            )
            print(f"Model '{local_model_name}' downloaded and saved locally.")
        else:
            print(f"Loading model from local path '{model_path}'...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_auth_token=HUGGINGFACE_TOKEN,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_auth_token=HUGGINGFACE_TOKEN,
            )
            print(f"Model '{local_model_name}' loaded from local storage.")
    else:
        print(f"Downloading model '{local_model_name}' without using local storage...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_name,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_auth_token=HUGGINGFACE_TOKEN,
        )
        print(f"Model '{local_model_name}' downloaded.")

    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline


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
# __main__ sanity check
# ---------------------------------------------------------
if __name__ == "__main__":
    # Sample question for sanity check
    sample_question = "What is the capital of France?"

    # Load the local model
    print("Loading local model...")
    generation_pipeline = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # Generate an answer
    print(f"Running sanity check with question: '{sample_question}'")
    local_answer = get_local_llm_answer(sample_question, generation_pipeline)
    print(f"Local LLM Answer: {local_answer}")
