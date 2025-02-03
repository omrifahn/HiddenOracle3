import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    HUGGINGFACE_TOKEN,
    LOCAL_MODEL_NAME,
    LOCAL_MODEL_DIR,
    USE_LOCAL_MODEL_STORAGE,
)

# Function to detect if running on Colab
def is_running_on_colab():
    return 'google.colab' in sys.modules

# Function to mount Google Drive
def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')

def load_local_model(local_model_name: str):
    """
    Loads the local LLM and returns a text generation pipeline, as well as
    the raw model and tokenizer.
    """
    if USE_LOCAL_MODEL_STORAGE:
        # If running on Colab, mount Google Drive
        if is_running_on_colab():
            print("Running on Google Colab. Mounting Google Drive...")
            mount_google_drive()
            print("Google Drive mounted.")

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
                torch_dtype=torch.float16,
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
                model_path, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                output_hidden_states=True,
                local_files_only=True,
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
            torch_dtype=torch.float16,
            use_auth_token=HUGGINGFACE_TOKEN,
            output_hidden_states=True,
        )
        print(f"Model '{local_model_name}' downloaded.")

    # Set up the generation pipeline without specifying the device
    generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer
    )
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
        answer = generated_text[len(question):].strip()
    else:
        answer = generated_text.strip()

    return answer


def get_local_llm_hidden_states(question: str, tokenizer, model, layer_index=20):
    """
    Returns the hidden states of the given layer_index for the input question.
    """
    # Tokenize the input
    inputs = tokenizer(question, return_tensors="pt")
    # Move inputs to the appropriate device
    if hasattr(model, 'hf_device_map'):
        # Get the device where the embeddings are placed
        embedding_device = next(iter(model.hf_device_map.values()))
        inputs = {key: value.to(embedding_device) for key, value in inputs.items()}
    else:
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass to get hidden states
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_index]
    return hidden_states
