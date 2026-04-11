import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

class FewShotType:
    RANDOM = "random"
    SIMILARITY = "similarity"
    SIZE_SAMPLE = "size_sample"

def few_shot(df: pd.DataFrame, type: FewShotType, n: int = 3):
    if type == FewShotType.RANDOM:
        return few_shot_random(df, n)
    elif type == FewShotType.SIMILARITY:
        return few_shot_similarity(df, n)
    elif type == FewShotType.SIZE_SAMPLE:
        return few_shot_size_sample(df, n)
    else:
        raise ValueError("Invalid few shot type")

def few_shot_random(df: pd.DataFrame, n: int= 3) -> list:
    # take n different random rows
    few_shot_samples: list = df.sample(n=n)
    # format to a list of dicts with only "prompt" and "answer"
    few_shots = []
    for _, row in few_shot_samples.iterrows():
        few_shots.append({
            "id":     row["id"],
            "prompt": row["Prompt"],
            "answer-1": row["Answer A (5)"],
            "rating-1": row["Rating"],
            "answer-2": row["Answer B (4-3)"],
            "rating-2": row["Rating.1"],
            "answer-3": row["Answer C (2-1)"],
            "rating-3": row["Rating.2"]
        })
    return few_shots

# TODO: Check later..
def few_shot_similarity(df: pd.DataFrame, n: int= 3) -> list:
    answers = df["Answer A (5)"].tolist()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(answers)
    
    min_similarity = float('inf')
    best_combination = None
    
    for combo in combinations(range(len(df)), n):
        combo_vectors = tfidf_matrix[list(combo)]
        similarities = cosine_similarity(combo_vectors)
        avg_similarity = (similarities.sum() - n) / (n * (n - 1))
        
        if avg_similarity < min_similarity:
            min_similarity = avg_similarity
            best_combination = combo
    
    few_shot_samples = df.iloc[list(best_combination)]
    few_shots = []
    for _, row in few_shot_samples.iterrows():
        few_shots.append({
            "id":     row["id"],
            "prompt": row["Prompt"],
            "answer-1": row["Answer A (5)"],
            "rating-1": row["Rating"],
            "answer-2": row["Answer B (4-3)"],
            "rating-2": row["Rating.1"],
            "answer-3": row["Answer C (2-1)"],
            "rating-3": row["Rating.2"]
        })
    return few_shots

def few_shot_size_sample(df: pd.DataFrame, n: int = 3) -> list:
    """Take equidistant samples basded on the lenght of the best answer"""
    answers = df["Answer A (5)"].tolist()
    answer_lengths = [(i, len(answer)) for i, answer in enumerate(answers)]
    answer_lengths.sort(key=lambda x: x[1])
    
    total = len(answer_lengths)
    selected_indices = []
    
    for i in range(n):
        position = int(i * (total - 1) / (n - 1)) if n > 1 else total // 2
        selected_indices.append(answer_lengths[position][0])
    
    few_shot_samples = df.iloc[selected_indices]
    few_shots = []
    for _, row in few_shot_samples.iterrows():
        few_shots.append({
            "id":     row["id"],
            "prompt": row["Prompt"],
            "answer-1": row["Answer A (5)"],
            "rating-1": row["Rating"],
            "answer-2": row["Answer B (4-3)"],
            "rating-2": row["Rating.1"],
            "answer-3": row["Answer C (2-1)"],
            "rating-3": row["Rating.2"]
        })
    return few_shots 

if __name__ == "__main__":
    df = pd.read_csv("csvs-train/Lexicology.csv")
    results: list = few_shot(df, FewShotType.SIZE_SAMPLE, 4)
    for result in results:
        print("ID: ", result["id"])
        print("Prompt: ", result["prompt"])
        print("Answer: ", result["answer-1"])
        print("-"*30)