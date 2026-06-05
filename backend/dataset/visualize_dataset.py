import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Try processed first, then raw
    csv_path = os.path.join(base_dir, "..", ".datasets", "reviews_labeled_processed.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(base_dir, "..", ".datasets", "reviews_labeled.csv")
    
    output_dir = os.path.join(base_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Dataset not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Pre-processing for visualization
    df["text_len_chars"] = df["text"].apply(lambda x: len(str(x)))
    df["text_len_words"] = df["text"].apply(lambda x: len(str(x).split()))
    
    def group_category(cat):
        cat_str = str(cat).lower()
        if cat_str in ["bnb", "restaurant"]:
            return cat_str
        return "amazon"
    
    df["grouped_category"] = df["category"].apply(group_category)

    # 1. Distribution of Insight Scores
    plt.figure()
    sns.histplot(df["insight_score"], kde=True, bins=30, color="skyblue")
    plt.title("Distribution of Insight Scores")
    plt.xlabel("Insight Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "insight_score_dist.png"))

    # 2. Insight Score by Grouped Category
    plt.figure()
    sns.boxplot(x="grouped_category", y="insight_score", data=df, palette="Set2")
    plt.title("Insight Score by Category")
    plt.xlabel("Category")
    plt.ylabel("Insight Score")
    plt.savefig(os.path.join(output_dir, "insight_score_by_cat.png"))

    # 3. Review Length (Words) Distribution
    plt.figure()
    sns.histplot(df[df["text_len_words"] < 500]["text_len_words"], kde=True, bins=50, color="orange")
    plt.title("Distribution of Review Length (Word Count < 500)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "review_length_dist.png"))

if __name__ == "__main__":
    main()
