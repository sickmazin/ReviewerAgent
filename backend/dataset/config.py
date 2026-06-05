# --- TEACHER CONFIGURATION ---
import os

MODEL_NAME = "gemma3:27b"
INPUT_AMAZON = "datasets/amazon_reviews_5M_random.csv"
INPUT_RESTAURANT = "datasets/Restaurant reviews.csv"
INPUT_BNB = "datasets/airbnb_reviews.csv"
OUTPUT_CSV = "datasets/reviews_labeled.csv"
INPUT_AMAZON = "../.datasets/amazon_reviews_5M_random.csv"
INPUT_RESTAURANT = "../.datasets/Restaurant reviews.csv"
INPUT_BNB = "../.datasets/airbnb_reviews.csv"
OUTPUT_CSV = "../.datasets/reviews_labeled.csv"
SAMPLE_SIZE = 500000

# --- AMAZON DATASET CONFIGURATION ---
REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
TOTAL_LIMIT = 5_000_000
OUTPUT_DIR = "../.datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "amazon_reviews_5M_random.csv")
