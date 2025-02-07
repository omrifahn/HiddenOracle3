# HiddenOracle: Single-Pass LLM + Classifier

## MIGHT NOT BE UPDATED, NOTICE THE DATE - February 7, 2025

**Environment**: Google Colab (A100 GPU)

This repository demonstrates how to:

1. **Generate answers** from a local Large Language Model (LLM).
2. **Determine factuality** (via a quick string match or OpenAI evaluation).
3. **Extract hidden states** from the LLM.
4. **Train a classifier** on those hidden states to predict hallucinations.

## Key Steps

1. **Install Requirements**\
   If you are running **locally**, use the provided    `requirements.txt` (e.g., `pip install -r requirements.txt`). On **Google Colab**, there's no need.

2. **Upload and Unzip**\
   In your Colab notebook:

   ```bash
   !unzip -q src.zip -d HiddenOracle3
   %cd HiddenOracle3/src
   ```

3. **Load and Precompute**

   For sanity check on 5 examples
     ```bash
     !python main.py
     ```
   And for the real expiriment:
     ```bash
     !python main.py 500
     ```
     This loads the model, **precomputes** LLM answers/hidden states in a single pass, and saves results to `output/output_data.json`.

4. **Train Classifier**

   - A simple linear classifier trains on cached features.
   - The code prints detailed logs, accuracy scores, and confusion breakdowns.


## TODOs
- connect to harder Q&A dataset (because currently, llama 90% right)



