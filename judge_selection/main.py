import pandas as pd
from choose_few_shot import few_shot, FewShotType
from prompts import format_prompt, PromptType
import os
from api import OpenAIClient, GeminiClient, DeepSeekClient, BedrockClient
import math
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from extracter import extract_response
from itertools import product
import time

def generous_diference_calc(predicted, expected) -> float:
    if expected == 5:
        return expected - predicted
    elif expected == 4 or expected == 3:
        if predicted >= 3 and predicted <= 4:
            return 0
        elif predicted < 3:
            return 3 - predicted
        elif predicted > 4:
            return predicted - 4
    elif expected == 2 or expected == 1:
        if predicted >= 1 and predicted <= 2:
            return 0
        elif predicted > 2:
            return predicted - 2
    print("ERRRRRRRRROOOOOOOOORRRRRRRR expected value: ", expected)
    raise ValueError("Invalid expected value")
        

def process_row(row, answer_col, rating_col, few_shot_examples, category, model_client, prompt_type):
    prompt = row["Prompt"]
    answer = row[answer_col]
    prompt = format_prompt(prompt_type, prompt, answer, few_shot_examples, category)
    waitTimeMultiplier = 5
    response = ''
    while True:
        try:
            response = model_client.chat(prompt)
            extracted_response = extract_response(prompt_type, response)
            result = {
                "prompt_id": row["id"],
                "category": category,
                "few_shot_ids": [ex["id"] for ex in few_shot_examples],
                "prompt": prompt,
                "answer": answer,
                "overall_score": extracted_response["overall_score"],
                "expecting_score": row[rating_col],
                "diference_score": math.fabs(row[rating_col] - extracted_response["overall_score"]),
                "generous_difference_score": generous_diference_calc(extracted_response["overall_score"], row[rating_col])
            }
            if prompt_type == PromptType.PROMPT_3_SCORES or prompt_type == PromptType.PROMPT_3_SCORES_PT:
                result['analyzes'] = {
                    "quality": {
                        "reasoning": extracted_response["quality_reasoning"],
                        "score": extracted_response["quality_score"]
                    },
                    "accuracy": {
                        "reasoning": extracted_response["accuracy_reasoning"],
                        "score": extracted_response["accuracy_score"]
                    },
                    "completeness": {
                        "reasoning": extracted_response["completeness_reasoning"],
                        "score": extracted_response["completeness_score"]
                    }
                },
            elif prompt_type == PromptType.PROMPT_1_SCORE or prompt_type == PromptType.PROMPT_1_SCORE_PT:
                result['reasoning'] = extracted_response["reasoning"],
            else:
                raise Exception("Invalid prompt type")
            break
        except Exception as e:
            print(f"\nError extracting response, retrying... {e}")
            # print response
            print(response)
            print("-"*30)
            time.sleep(waitTimeMultiplier)
            waitTimeMultiplier *= 2
            continue
    
    

    return result

def process_csv(csv_file: str, results_folder: str, model_client, few_shot_strategy, n_few_shots, prompt_type):
    df = pd.read_csv(csv_file)
    category = os.path.basename(csv_file).split(".")[0]
    few_shot_examples: list = few_shot(df, few_shot_strategy, n_few_shots)
    few_shot_ids = [ex["id"] for ex in few_shot_examples]
    df = df[~df["id"].isin(few_shot_ids)]
    answers_cols = ["Answer A (5)", "Answer B (4-3)", "Answer C (2-1)"]
    rating_cols = ["Rating", "Rating.1", "Rating.2"]
    tasks = [(row, answer_col, rating_col, few_shot_examples, category, model_client, prompt_type) 
             for _, row in df.iterrows() 
             for answer_col, rating_col in zip(answers_cols, rating_cols)]    
    workers = 5 if model_client.model.startswith("gemini") else 10
    with ThreadPoolExecutor(max_workers=workers) as executor:
        all_results = list(executor.map(lambda t: process_row(*t), tasks))
    
    with open(f"{results_folder}/{category}.json", "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)


def get_generous_score(expected_score: int) -> str:
    if expected_score == 5:
        return "5"
    if expected_score == 4 or expected_score == 3:
        return "3-4"
    elif expected_score == 2 or expected_score == 1:
        return "1-2"
    raise ValueError("Invalid expected value")

def make_result_stats(results_folder: str, model_client, few_shot_strategy, n_few_shots, prompt_type, update: bool = False):
    # Making stats...
    print("Making stats...")
    if not update and os.path.exists(f"{results_folder}/Z-Stats.json"):
        return None
    results = {}
    if update:
        file = f"{results_folder}/Z-Stats.json"
        with open(file, "r") as f:
            results = json.load(f)
    else:
        results['settings'] = {
            'model': model_client.model,
            'prompt_type': prompt_type,
            'few_shot_strategy': few_shot_strategy,
            'few_shot_n': n_few_shots,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    all_generous_diferences = []
    all_differences = []
    generous_differences_per_rating = {
        "1-2": [],
        "3-4": [],
        "5": []
    }
    differences_per_rating = {i: [] for i in range(1, 6)}
    for file in os.listdir(results_folder):
        if file.endswith(".json") and not file.startswith("Z-Stats"):
            with open(f"{results_folder}/{file}", "r") as f:
                file_results = json.load(f)
                all_differences_file = []
                all_generous_diferences_file = []
                differences_per_rating_file = {i: [] for i in range(1, 6)}
                generous_differences_per_rating_file = {
                    "1-2": [],
                    "3-4": [],
                    "5": []
                }
                for result in file_results:
                    
                    all_differences_file.append(result["diference_score"])
                    all_differences.append(result["diference_score"])
                    differences_per_rating_file[int(result["expecting_score"])].append(result["diference_score"])
                    differences_per_rating[int(result["expecting_score"])].append(result["diference_score"])

                    all_generous_diferences_file.append(result["generous_difference_score"])
                    all_generous_diferences.append(result["generous_difference_score"])
                    generous_differences_per_rating_file[get_generous_score(result["expecting_score"])].append(result["generous_difference_score"])
                    generous_differences_per_rating[get_generous_score(result["expecting_score"])].append(result["generous_difference_score"])

                for rating in differences_per_rating_file:
                    differences_per_rating_file[rating] = sum(differences_per_rating_file[rating]) / len(differences_per_rating_file[rating]) if len(differences_per_rating_file[rating]) > 0 else 0
                for rating in generous_differences_per_rating_file:
                    rating: str = rating
                    generous_differences_per_rating_file[rating] = sum(generous_differences_per_rating_file[rating]) / len(generous_differences_per_rating_file[rating]) if len(generous_differences_per_rating_file[rating]) > 0 else 0
                
                cat = file.rsplit('.json')[0]
                results[cat] = {
                    'difference_avg': sum(all_differences_file) / len(all_differences_file),
                    'difference_max': max(all_differences_file),
                    'difference_min': min(all_differences_file),
                    'difference_median': sorted(all_differences_file)[len(all_differences_file) // 2],
                    'difference_std': (sum([(x - sum(all_differences_file) / len(all_differences_file)) ** 2 for x in all_differences_file]) / len(all_differences_file)) ** 0.5,
                    'generous_difference_avg': sum(all_generous_diferences_file) / len(all_generous_diferences_file),
                    'generous_difference_max': max(all_generous_diferences_file),
                    'generous_difference_min': min(all_generous_diferences_file),
                    'generous_difference_median': sorted(all_generous_diferences_file)[len(all_generous_diferences_file) // 2],
                    'generous_difference_std': (sum([(x - sum(all_generous_diferences_file) / len(all_generous_diferences_file)) ** 2 for x in all_generous_diferences_file]) / len(all_generous_diferences_file)) ** 0.5
                }
                results[cat]['difference_avg_per_rating'] = {}
                for rating in differences_per_rating_file:
                    results[cat]['difference_avg_per_rating'][f'difference_avg_{rating}'] = differences_per_rating_file[rating]
                
                results[cat]['generous_difference_avg_per_rating'] = {}
                for rating in generous_differences_per_rating_file:
                    results[cat]['generous_difference_avg_per_rating'][f'generous_difference_avg_{rating}'] = generous_differences_per_rating_file[rating]

    results['all'] = {
        'difference_avg': sum(all_differences) / len(all_differences),
        'difference_max': max(all_differences),
        'difference_min': min(all_differences),
        'difference_median': sorted(all_differences)[len(all_differences) // 2],
        'difference_std': (sum([(x - sum(all_differences) / len(all_differences)) ** 2 for x in all_differences]) / len(all_differences)) ** 0.5,
        'generous_difference_avg': sum(all_generous_diferences) / len(all_generous_diferences),
        'generous_difference_max': max(all_generous_diferences),
        'generous_difference_min': min(all_generous_diferences),
        'generous_difference_median': sorted(all_generous_diferences)[len(all_generous_diferences) // 2],
        'generous_difference_std': (sum([(x - sum(all_generous_diferences) / len(all_generous_diferences)) ** 2 for x in all_generous_diferences]) / len(all_generous_diferences)) ** 0.5
    }
    results['all']['difference_avg_per_rating'] = {}
    for rating in differences_per_rating:
        results['all']['difference_avg_per_rating'][f'difference_avg_{rating}'] = sum(differences_per_rating[rating]) / len(differences_per_rating[rating]) if len(differences_per_rating[rating]) > 0 else 0

    results['all']['generous_difference_avg_per_rating'] = {}
    for rating in generous_differences_per_rating:
        rating: str = rating
        results['all']['generous_difference_avg_per_rating'][f'generous_difference_avg_{rating}'] = sum(generous_differences_per_rating[rating]) / len(generous_differences_per_rating[rating]) if len(generous_differences_per_rating[rating]) > 0 else 0

    with open(f"{results_folder}/Z-Stats.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def get_all_csvs(folder: str) -> list:
    csvs = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            csvs.append(os.path.join(folder, file))
    return csvs

def run_model_configurations(model, strategies, ns, prompt_types, csvs, pbar, lock, results_folder):
    model_name = model.model
    skip = False
    # get all folders on "results"
    folders = [f for f in os.listdir(results_folder)]
    for strategy in strategies:
        for n in ns:
            for prompt_type in prompt_types:
                if "/" in model_name:
                    model_name = model_name.split("/")[-1]
                folder_name = f"{model_name}_{strategy}_{n}_{prompt_type}"
                print(f"Folder name: {folder_name}")
                for folder in folders:
                    if folder.startswith(folder_name):
                        print(f"Skipping {folder}")
                        with lock:
                            pbar.update(1)
                        skip = True
                        break
                if skip:
                    skip = False
                    continue

                run_results_folder = f"{results_folder}/{model.model}_{strategy}_{n}_{prompt_type}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
                os.makedirs(run_results_folder, exist_ok=True)
                
                with lock:
                    pbar.set_description(f"{model.model} | {strategy} | {n} shots | {prompt_type}")
                
                for csv in csvs:
                    process_csv(csv, run_results_folder, model, strategy, n, prompt_type)
                make_result_stats(run_results_folder, model, strategy, n, prompt_type)
                
                with lock:
                    pbar.update(1)

def delete_redundant_folders(results_folder: str):
    os.makedirs(results_folder, exist_ok=True)
    # look at results folder
    folders = [f for f in os.listdir(results_folder)]
    # if a folder does not have the file "Z-Stats.json" delete it
    for folder in folders:
        if not os.path.exists(f"{results_folder}/{folder}/Z-Stats.json"):
            print(f"Deleting {folder}")
            os.system(f"rm -rf {results_folder}/{folder}")


def main(results_folder: str, csv_folder: str):
    delete_redundant_folders(results_folder)
    from threading import Lock
    
    models = [BedrockClient("openai.gpt-oss-120b-1:0"), BedrockClient("openai.gpt-oss-safeguard-120b")] # OpenAIClient() GeminiClient() DeepseekClient()
    few_shot_strategies = [FewShotType.RANDOM, FewShotType.SIMILARITY, FewShotType.SIZE_SAMPLE]
    few_shot_ns = [2, 3, 4, 5]
    prompt_types = [PromptType.PROMPT_3_SCORES, PromptType.PROMPT_1_SCORE, PromptType.PROMPT_3_SCORES_PT, PromptType.PROMPT_1_SCORE_PT]
    
    csvs = get_all_csvs(csv_folder)
    
    total_iterations = len(models) * len(few_shot_strategies) * len(few_shot_ns) * len(prompt_types)
    lock = Lock()
    
    with tqdm(total=total_iterations, desc="Progress") as pbar:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(run_model_configurations, model, few_shot_strategies, few_shot_ns, prompt_types, csvs, pbar, lock, results_folder) for model in models]
            for future in futures:
                future.result()

def run_evaluation(few_shot_strategies, few_shot_ns, prompt_types, models, split_folder, results_folder):
    delete_redundant_folders(results_folder)
    from threading import Lock
    
    csvs = get_all_csvs(split_folder)
    total_iterations = len(models) * len(few_shot_strategies) * len(few_shot_ns) * len(prompt_types)
    lock = Lock()
    
    with tqdm(total=total_iterations, desc="Progress") as pbar:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = [executor.submit(run_model_configurations, model, few_shot_strategies, few_shot_ns, prompt_types, csvs, pbar, lock, results_folder) for model in models]
            for future in futures:
                future.result()

def update_all_zScores(results_folder: str):
    for folder in os.listdir(results_folder):
        if os.path.isdir(os.path.join(results_folder, folder)):
            make_result_stats(os.path.join(results_folder, folder), None, None, 0, None, update=True)

def update_all_ratings(results_folder: str):
    # For all folders inside
    folders = [f for f in os.listdir(results_folder)]
    # For all files inside the folder not named Z-Stats.json
    for folder in tqdm(folders):
        if os.path.isdir(os.path.join(results_folder, folder)):
            for file in os.listdir(os.path.join(results_folder, folder)):
                if file.endswith("Z-Stats.json"):
                  continue  
                # Open the json file
                with open(os.path.join(results_folder, folder, file), "r") as f:
                    ratings: list[dict] = json.load(f)
                    # For each element, add a key "generous_difference_score"
                    for elem in ratings:
                        elem["generous_difference_score"] = generous_diference_calc(elem["overall_score"], elem["expecting_score"])
                with open(os.path.join(results_folder, folder, file), "w") as f2:
                    json.dump(ratings, f2, indent=4, ensure_ascii=False)



def find_best_judge(results_folder: str):
    run_evaluation(
        few_shot_strategies = [FewShotType.SIZE_SAMPLE],
        few_shot_ns         = [3],
        prompt_types        = [PromptType.PROMPT_1_SCORE_PT],
        models = [
            #GeminiClient(), 
            #OpenAIClient(), 
            #DeepSeekClient(), 
            #GeminiClient("gemini-2.5-flash"), 
            #OpenAIClient("gpt-5.1-2025-11-13"), 
            #DeepSeekClient("deepseek-reasoner"),
            #BedrockClient("us.amazon.nova-premier-v1:0"), 
            #BedrockClient("cohere.command-r-plus-v1:0"),
            BedrockClient("openai.gpt-oss-120b-1:0"), 
            BedrockClient("openai.gpt-oss-safeguard-120b")
        ],
        split_folder = 'judge_selection/csvs-train',
        results_folder = results_folder
    )

def find_optimal_config(results_folder: str):
    run_evaluation(
        few_shot_strategies = [FewShotType.RANDOM, FewShotType.SIMILARITY, FewShotType.SIZE_SAMPLE],
        few_shot_ns = [2, 3, 4, 5],
        prompt_types = [PromptType.PROMPT_3_SCORES, PromptType.PROMPT_1_SCORE, PromptType.PROMPT_3_SCORES_PT, PromptType.PROMPT_1_SCORE_PT],
        models = [
            GeminiClient(), 
            OpenAIClient(), 
            DeepSeekClient(), 
            GeminiClient("gemini-2.5-flash"), 
            OpenAIClient("gpt-5.1-2025-11-13"), 
            DeepSeekClient("deepseek-reasoner"),
            BedrockClient("us.amazon.nova-premier-v1:0"), 
            BedrockClient("cohere.command-r-plus-v1:0"),
            BedrockClient("openai.gpt-oss-120b-1:0"), 
            BedrockClient("openai.gpt-oss-safeguard-120b")
        ],
        split_folder = 'judge_selection/csvs-test',
        results_folder = results_folder
    )

if __name__ == "__main__":
    results_folder = "judge_selection/results-test"
    # update_all_zScores(results_folder)
    # main(results_folder, "judge_selection/csvs-test")

    # find_optimal_config(results_folder)
    find_best_judge(results_folder)
