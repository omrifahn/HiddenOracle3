from config import DATASET_PATH, LOCAL_MODEL_NAME
from evaluator import evaluate_model

if __name__ == "__main__":
    evaluate_model(DATASET_PATH, LOCAL_MODEL_NAME)
