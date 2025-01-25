from config import DATASET_PATH, LOCAL_MODEL_NAME
from evaluator import evaluate_model

# ---------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    # Run evaluation
    evaluate_model(DATASET_PATH, LOCAL_MODEL_NAME)
