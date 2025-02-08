# HiddenOracle: Single-Pass LLM + Classifier

**NOTICE: Updated February 8, 2025**

**Environment:** Google Colab (A100 GPU) or local GPU/CPU setups

HiddenOracle now features a three-part modular pipeline that:

1. **Creates a Dataset**  
   Samples TriviaQA to produce `triviaqa_data.json`.
2. **Extracts Hidden States & Evaluates Answers**  
   Uses a local LLM (default: `meta-llama/Llama-2-7b-chat-hf`) to generate answers, evaluates factuality (via string matching and OpenAI API), and extracts hidden states.
3. **Trains a Classifier**  
   Trains a Logistic Regression classifier on the extracted hidden states to predict hallucinations.

---

## Key Steps

1. **Install Requirements**  
   If running locally, use the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   On Google Colab, most packages are pre-installed.

2. **Dataset Creation**  
   Generate the TriviaQA dataset (500 datapoints by default) by running:

   ```bash
   python create_dataset/create_dataset.py
   ```

   This produces `triviaqa_data.json`.

3. **Hidden State Extraction & Evaluation**  
   Process the dataset using the local LLM to generate answers, evaluate factuality, extract hidden states, and produce a detailed run report:

   ```bash
   python hidden_state_extraction/hidden_state_extraction.py
   ```

   _Note:_ Runtime statistics, error tracking, and log details (with sample IDs) are now included. Dataset and output paths are relative to the config file.

4. **Classifier Training**  
   Train a Logistic Regression classifier on the enriched data to predict factual vs. hallucinated answers:
   ```bash
   python classifier_training/train_classifier.py
   ```
   The script displays data balance, accuracy, and a confusion matrix, and saves a run report.
