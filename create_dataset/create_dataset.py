import json
import random
from datasets import load_dataset

NUM_SAMPLES = 500
OUTPUT_FILE = "triviaqa_data.json"


def load_triviaqa(num_samples):
    dataset = load_dataset("trivia_qa", "unfiltered", split="train", streaming=True)
    samples = []
    keep_probability = 0.05  # for random sampling
    last_logged_count = 0

    for idx, item in enumerate(dataset):
        if random.random() < keep_probability:
            samples.append(item)
            if len(samples) % 10 == 0 and len(samples) != last_logged_count:
                print(f"Collected {len(samples)}/{num_samples} samples...")
                last_logged_count = len(samples)
            if len(samples) >= num_samples:
                break
    return samples


def process_triviaqa_data(data):
    processed_data = []
    for idx, item in enumerate(data):
        question = item.get("question", "")
        answer_data = item.get("answer", {})
        aliases = answer_data.get("aliases", [])
        normalized_aliases = answer_data.get("normalized_aliases", [])
        all_answers = list(set(aliases + normalized_aliases))
        processed_data.append(
            {"id": idx + 1, "question": question, "answers": all_answers}
        )
    return processed_data


def save_processed_data(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Processed TriviaQA data saved to {OUTPUT_FILE}")


def main():
    data = load_triviaqa(NUM_SAMPLES)
    processed_data = process_triviaqa_data(data)
    save_processed_data(processed_data)


if __name__ == "__main__":
    main()
