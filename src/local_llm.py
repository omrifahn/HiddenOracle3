import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    HUGGINGFACE_TOKEN,
    LOCAL_MODEL_DIR,
    USE_LOCAL_MODEL_STORAGE,
)


# Function to detect if running on Colab
def is_running_on_colab():
    return "google.colab" in sys.modules


# Function to mount Google Drive if on Colab
def mount_google_drive():
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def load_local_model(local_model_name: str):
    """
    Loads the local LLM and returns a text generation pipeline,
    as well as the raw model and tokenizer.
    """
    if USE_LOCAL_MODEL_STORAGE:
        # If running on Colab, mount Google Drive
        if is_running_on_colab():
            print("Running on Google Colab. Mounting Google Drive...")
            mount_google_drive()
            print("Google Drive mounted.")

        # Replace slashes to create a valid directory name
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
            # Model exists locally, so just load it
            print(f"Loading model from local path '{model_path}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
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

    # Set up the generation pipeline
    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline, model, tokenizer


def get_local_llm_answer(question: str, model, tokenizer, max_new_tokens=80):
    """
    Example for the 'meta-llama/Llama-2-7b-chat-hf' model
    using the recommended [INST] ... [/INST] format.
    """
    system_prompt = (
        "You are a helpful assistant. Provide a single short factual answer. "
        "Do not continue the conversation or ask follow-up questions."
    )

    # Format the prompt per Llama 2 Chat guidelines
    prompt = f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

{question}
[/INST]"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False,
        )

    # Decode
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip off the prompt part if needed
    if "[INST]" in text:
        # Typically we split at the last "[/INST]"
        text = text.split("[/INST]")[-1]
    return text.strip()


def get_local_llm_hidden_states(question: str, tokenizer, model, layer_index=20):
    """
    Returns the hidden states of the given layer_index for the input question.
    """
    # Tokenize the input
    inputs = tokenizer(question, return_tensors="pt")
    # Move inputs to the appropriate device
    if hasattr(model, "hf_device_map"):
        # For models with device_map
        embedding_device = next(iter(model.hf_device_map.values()))
        inputs = {key: value.to(embedding_device) for key, value in inputs.items()}
    else:
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass to get hidden states
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_index]
    return hidden_states
