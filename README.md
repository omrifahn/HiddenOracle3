# Hidden Oracle 3

**Hidden Oracle 3** is a research project aimed at exploring the internal representations of large language models (LLMs) to assess the factuality of their responses. By examining the hidden states of a local LLM while it answers trivia questions, this project investigates correlations between internal activations and answer reliability. The ultimate goal is to shed light on how model internals relate to factual accuracy and hallucination tendencies.

---

## Overview

The project follows an end-to-end pipeline that spans multiple research components:

- **Dataset Creation:**  
  A subset of the TriviaQA dataset is sampled and pre-processed. Questions and their corresponding multiple correct answers are extracted to serve as the foundation for further analysis.

- **Hidden State Extraction & Enrichment:**  
  Using a local language model (Llama-2-7b-chat), each question is answered. The project then:
  - **Evaluates** the generated answer for factual correctness using both simple string matching and a secondary evaluation via the OpenAI API.
  - **Extracts** hidden state vectors from the model (from one or multiple layers) for each question. These vectors serve as features to understand model behavior with respect to factuality.

- **Data Balancing:**  
  To ensure a robust evaluation, the dataset is balanced by undersampling the majority class. This step guarantees an even distribution of factual versus hallucinated samples.

- **Classifier Training:**  
  Logistic regression classifiers are trained on the extracted hidden state vectors, one per layer, to predict the factuality of the model's responses. Performance metrics (accuracy, precision, recall, etc.) are computed to analyze which layers contribute most significantly to factuality detection.

---

## Research Motivation & Contributions

- **Interpretability:**  
  By linking hidden state representations to factual correctness, this project contributes to our understanding of the inner workings of LLMs and how they encode information related to truthfulness.

- **Factuality Assessment:**  
  The dual approach of simple string matching and API-assisted evaluation provides a nuanced method to distinguish between factual responses and hallucinations.

- **Layer-wise Analysis:**  
  Training classifiers on different layers’ hidden states allows for an investigation into which parts of the model are most predictive of factual accuracy. This insight can inform future work on model design and debugging.

- **Balanced Evaluation:**  
  Ensuring a balanced dataset allows for more rigorous evaluation of classifier performance and reduces bias stemming from class imbalances.

---

## Project Structure

- **Dataset Creation:**  
  - *Module:* `create_dataset.py`  
    Extracts and processes a random subset of TriviaQA data.

- **Hidden State Extraction & Enrichment:**  
  - *Module:* `hidden_state_extraction/`  
    Orchestrates local model answer generation, evaluation, and hidden state extraction.
  - *Key Components:*  
    - **Local LLM Interface:** Loads the model, generates answers, and retrieves hidden states.
    - **Evaluation:** Combines simple string matching with an OpenAI API-based evaluation to judge answer factuality.

- **Data Balancing:**  
  - *Module:* `data_balancing/data_balancing.py`  
    Balances the dataset by undersampling the dominant class to create an equal representation of factual and hallucinated samples.

- **Classifier Training:**  
  - *Module:* `classifier_training/train_classifier.py`  
    Trains logistic regression classifiers on hidden state features from each model layer, providing insights into layer-wise predictive power regarding factuality.

- **Configuration & Utilities:**  
  Common configuration files and helper scripts support consistent data paths, model handling, and logging across the project.

---

## Conclusion

**Hidden Oracle 3** serves as a framework for probing the internal representations of LLMs, aiming to link hidden state dynamics with factual accuracy. The insights derived from this research could pave the way for improved interpretability, more reliable fact-checking methods, and enhanced design of future language models.

For further details on methodology and results, please refer to the associated research publications and supplementary materials.

