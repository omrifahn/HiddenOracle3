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
    Loads the local LLM and returns a text generation pipeline, as well as
    the raw model and tokenizer (needed to extract hidden states).
    If USE_LOCAL_MODEL_STORAGE is True and the model is already downloaded
    in LOCAL_MODEL_DIR, it loads from storage. Otherwise, it downloads the
    model and saves it locally to save internet downloads in the future.
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
                local_model_name, use_auth_token=HUGGINGFACE_TOKEN
            )
            model = AutoModelForCausalLM.from_pretrained(
                local_model_name,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_auth_token=HUGGINGFACE_TOKEN,
                output_hidden_states=True,
            )
            # Save the model and tokenizer to the local directory
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print(f"Model '{local_model_name}' downloaded and saved to '{model_path}'.")
        else:
            # Model exists locally. Load it without authentication.
            print(f"Loading model from local path '{model_path}'...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True  # Ensure loading from local files
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                output_hidden_states=True,
                local_files_only=True,  # Ensure loading from local files
            )
            print(f"Model '{local_model_name}' loaded from local storage.")
    else:
        # Download model without using local storage
        print(f"Downloading model '{local_model_name}' without using local storage...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_name, use_auth_token=HUGGINGFACE_TOKEN
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_auth_token=HUGGINGFACE_TOKEN,
            output_hidden_states=True,
        )
        print(f"Model '{local_model_name}' downloaded.")

    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline, model, tokenizer


def get_local_llm_answer(question: str, generation_pipeline) -> str:
    """
    Uses a text-generation pipeline to get an answer from the local LLM.
    """
    results = generation_pipeline(
        question, max_new_tokens=25, num_return_sequences=1, do_sample=False
    )
    generated_text = results[0]["generated_text"]

    if generated_text.startswith(question):
        answer = generated_text[len(question) :].strip()
    else:
        answer = generated_text.strip()

    return answer


def get_local_llm_hidden_states(question: str, tokenizer, model, layer_index=20):
    """
    Returns the hidden states of the given layer_index for the input question.
    By default, extracts layer 20 of the Llama model.

    :param question: The input text to encode.
    :param tokenizer: Tokenizer associated with the model.
    :param model: The Llama (or any HF) model with output_hidden_states=True.
    :param layer_index: The hidden layer index to return (0-based).
    :return: A torch.Tensor of shape (batch_size, seq_len, hidden_size).
    """
    # Tokenize the input
    inputs = tokenizer(question, return_tensors="pt")
    # Forward pass to get hidden states
    outputs = model(**inputs)
    # `hidden_states` is a tuple of length [n_layers + 1], each of shape (batch_size, seq_len, hidden_dim)
    # layer_index=0 corresponds to the embeddings, so typically "layer 20" is hidden_states[20]
    hidden_states = outputs.hidden_states[layer_index]
    return hidden_states


# ---------------------------------------------------------
# __main__ sanity check
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_question = "What is the capital of France?"
    print("Loading local model...")
    generation_pipeline, model, tokenizer = load_local_model(LOCAL_MODEL_NAME)
    print("Local model loaded.")

    # Test text generation
    print(f"Running sanity check with question: '{sample_question}'")
    local_answer = get_local_llm_answer(sample_question, generation_pipeline)
    print(f"Local LLM Answer: {local_answer}")

    # Test hidden state extraction
    layer_20_states = get_local_llm_hidden_states(
        sample_question, tokenizer, model, layer_index=20
    )
    print(f"Hidden States (Layer 20) shape: {layer_20_states.shape}")
