import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import HUGGINGFACE_TOKEN, LOCAL_MODEL_DIR, USE_LOCAL_MODEL_STORAGE


def is_running_on_colab():
    return "google.colab" in sys.modules


def mount_google_drive():
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def load_local_model(local_model_name: str):
    print(f"[DEBUG] USE_LOCAL_MODEL_STORAGE = {USE_LOCAL_MODEL_STORAGE}")
    print(f"[DEBUG] local_model_name = {local_model_name}")
    model_dir_name = local_model_name.replace("/", "_")
    model_path = os.path.join(LOCAL_MODEL_DIR, model_dir_name)
    print(f"[DEBUG] model_path = {model_path}")

    if USE_LOCAL_MODEL_STORAGE:
        if is_running_on_colab():
            print("[DEBUG] Running on Google Colab. Mounting Google Drive...")
            mount_google_drive()
            print("[DEBUG] Google Drive mounted.")

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            print(
                f"[DEBUG] Model not found locally at {model_path}. Downloading '{local_model_name}'..."
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
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print(
                f"[DEBUG] Model '{local_model_name}' downloaded and saved to '{model_path}'."
            )
        else:
            print(
                f"[DEBUG] Found existing model locally at {model_path}. Loading it now..."
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                output_hidden_states=True,
                local_files_only=True,
            )
            print(f"[DEBUG] Model '{local_model_name}' loaded from local storage.")
    else:
        print(
            f"[DEBUG] USE_LOCAL_MODEL_STORAGE = False. Downloading model '{local_model_name}' from HuggingFace..."
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
        print(f"[DEBUG] Model '{local_model_name}' downloaded (no local storage).")

    generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generation_pipeline, model, tokenizer


def get_local_llm_answer(question: str, model, tokenizer, max_new_tokens=80):
    system_prompt = (
        "You are a helpful assistant. Provide a single short factual answer. "
        "Do not continue the conversation or ask follow-up questions."
    )
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
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[INST]" in text:
        text = text.split("[/INST]")[-1]
    return text.strip()


def get_local_llm_hidden_states(question: str, tokenizer, model, layer_index=20):
    inputs = tokenizer(question, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states[layer_index]
    return hidden_states
