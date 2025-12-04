#!/usr/bin/env python3

from dataclasses import asdict, dataclass
import os, sys, json, time, tyro
import numpy as np
import pandas as pd

from typing import Any, Union, Annotated
from dotenv import load_dotenv
from glob import glob

from models import Gemini, HuggingFaceModel, Model
from scorer import Metric, score_samples, SamplePair
from typing import cast

def extract_bench_id(data : dict[str, Any]) -> str:
    keys = list(data['results'].keys())
    assert len(keys) == 1
    return keys[0]

def get_bench_samples_filename(bench_id : str, model_dir : str) -> str:
    sample_files = glob(f'{model_dir}/samples_{bench_id}*.jsonl')
    assert len(sample_files) == 1, f"found more than 1 samples file for '{bench_id}': {','.join(sample_files)}"
    return sample_files[0]

def rescore_sample(sample : dict, score : Metric) -> dict:
    values = asdict(score)
    new_sample = sample.copy() 
    new_sample.pop('bypass')

    return new_sample | { 'metrics': list(values.keys()) } | values

def fix_log_data(log_data : dict[str, Any], bench_id : str, scores : list[Metric]):
    mean, stderr = aggregate_scores(scores)

    # FIXME: find a way to make this not be this error prone
    log_data['results'][bench_id] |= {
      "score,none": mean,
      "score_stderr,none": stderr,
      "explanation,none": 0,
      "explanation_stderr,none": "N/A",
      "judge_prompt,none": 0,
      "judge_prompt_stderr,none": "N/A",
      "judge_plain_answer,none": 0,
      "judge_plain_answer_stderr,none": "N/A"
    } 

    log_data['results'][bench_id].pop('bypass,none')
    log_data['results'][bench_id].pop('bypass_stderr,none')

    log_data['configs'][bench_id]['metric_list'] = [
        {
          "metric": "score",
          "aggregation": "mean",
          "higher_is_better": True
        },
        {
          "metric": "explanation",
          "aggregation": "def plain(_): return 0\n",
          "higher_is_better": True
        },
        {
          "metric": "judge_prompt",
          "aggregation": "def plain(_): return 0\n",
          "higher_is_better": True
        },
        {
          "metric": "judge_plain_answer",
          "aggregation": "def plain(_): return 0\n",
          "higher_is_better": True
        }
    ] 

    log_data['higher_is_better'] = {
        bench_id: {
          "score": True,
          "explanation": True,
          "judge_prompt": True,
          "judge_plain_answer": True,
        }
    }
    
def aggregate_scores(scores : list[Metric]) -> tuple[tuple, tuple]:
    plain_scores = [score.score for score in scores]
    return np.mean(plain_scores), np.std(plain_scores) # pyright: ignore


def legacy_score_models(models_dir : str, judge : Model): 
    print(f"> processing models samples in '{models_dir}'")
    for model_dir in glob(f'{models_dir}/*'):
        log_files = glob(f'{model_dir}/*.json')

        if len(log_files) == 0:
            print(f"> skipping '{model_dir}' ...")
            continue

        print(f"> processing '{model_dir}' ...")
        for log_filename in log_files:
            print(f">> processing log file '{log_filename}'")
            with open(log_filename) as file:
                log_data = json.load(file)

            bench_id = extract_bench_id(log_data)
            print(f'>>> bench_id: {bench_id}')

            if len(log_data['configs'][bench_id]['metric_list']) > 1:
                print('>> skipping log file (it was most probably already rescored) ...')
                continue

            samples_filename = get_bench_samples_filename(bench_id, model_dir)
            print(f'>>> loading sample file: {samples_filename}')

            with open(samples_filename) as file:
                samples = [json.loads(sample) for sample in file]

            category = samples[0]['doc']['category']

            print(f'>>> category: {category}')
            assert all(map(lambda entry: entry['doc']['category'] == category, samples))
            assert all(map(lambda entry: len(entry['resps']) == 1, samples))
            assert all(map(lambda entry: len(entry['resps'][0]) == 1, samples))
            assert all(map(lambda entry: len(entry['filtered_resps']) == 1, samples))
            assert all(map(lambda entry: entry['resps'][0] == entry['filtered_resps'], samples))

            print('>>> generating scores')
            scores = score_samples(
                judge, 
                [
                    SamplePair(
                        prompt = sample['doc']['prompt'],
                        response = sample['resps'][0][0].split('</think>')[-1].strip() # remove think tokens for qwen guard
                    )
                    for sample in samples
                ], # TODO: should I do strip?
                category,
                max_connections = 50
            )

            # the following 2 instructions tries to format the samples and log_data as if the whole pipeline was ran on 
            # lm-evaluation-harness, the purpose of this is to try to be able to use the their display tool for debug and
            # have our scripts that already take results from lm-evaluation-harness work on these ones as well
            new_samples = [ rescore_sample(sample, score) for sample, score in zip(samples, scores) ]
            fix_log_data(log_data, bench_id, scores) # TODO: think if you want to copy or just alter the current one

            # TODO: SHOULD YOU JUST REPLACE THE FILES OR SHOULD YOU DO A COPY OF THEM
            print('>>> writing new version of the log file and samples file')
            with open(log_filename, 'w') as file:
                json.dump(log_data, file, indent=4, ensure_ascii=False)

            with open(samples_filename, 'w') as file:
                file.writelines([
                    json.dumps(sample, ensure_ascii=False) + '\n' for sample in new_samples
                ])

def extract_model_name(model_name : str) -> str:
    parts = model_name.split('/')
    if not parts[-1]: parts.pop()
    return '/'.join(parts[-2:]) if parts[-1].startswith('checkpoint') else parts[-1]

RENAMES = {
    "47-32k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-2875": "AMALIA-9B 32k v49",
    "47-4k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-13590": "AMALIA-9B 4k v49",
    "47-32k-llama/checkpoint-700": "AMALIA-LLaMA-3.1-8B-32k",
    "49-32k-llama_instruct/checkpoint-1767": "AMALIA-LLaMA-3.1-8B-Instruct-32k",
    "47-32k-qwen3_8B/checkpoint-1482": "AMALIA-Qwen3-8B-32k",
    "49-32k-eurollm-9B/checkpoint-1928": "EuroLLM-AMALIA-9B-32k v49 (checkpoint 1928)",
    "49-32k-gemma3-12B/checkpoint-1368": "AMALIA-Gemma3-12B-32k",
    "47-safety-dpo-mix_safety_sft_200k/checkpoint-6738_merged": "49 DPO",
    "50-carminho-big/checkpoint-1776": "AMALIA-9B-32k-big v50 (checkpoint 1776)", # 32k SFT-BIG
    "50-carminho-big/checkpoint-3441": "AMALIA-9B-32k-big v50 (checkpoint 3441)", # 32k SFT-BIG
    "50-carminho-big/checkpoint-3480": "AMALIA-9B-32k-big v50 (checkpoint 3480)",
    "50-dpo-mix_safety_sft_200k_if/checkpoint-6892_merged": "AMALIA-9B-32k-big-DPO-small v50",
    "50-carminho-big-old/checkpoint-18501":  "AMALIA-9B-4k-big v50",
    "50-big-4k-dpo-big/checkpoint-6155_merged": "AMALIA-9B-4k-big-DPO-big v50",
    "49-4k-eurollm-9B/checkpoint-12231": "EuroLLM AMALIA-9B 4k v49 (checkpoint 12231)",
    "Llama-3.1-8B-Instruct": "LLaMA 3.1-Instruct-8B",
    "Ministral-8B-Instruct-2410": "Ministral-8B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "OLMo-2-1124-7B-Instruct": "OLMo 2-7B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
    "gervasio-8b-portuguese-ptpt-decoder": "LLaMA-3.1-Gervasio-8B",
    "salamandra-7b-instruct": "Salamandra-7B",
}

def new_score_models(models_log_files : list[str], judge : Model):
    for file in models_log_files:

        if file.endswith('_scored.csv'): continue

        print(f"> loading '{file}'")
        out_file = f'{file.rstrip('.csv')}_scored.csv'

        if os.path.isfile(out_file):
            print('>> skipping probably already scored ...')
            continue


        data = pd.read_csv(file)
        model_name      =  data.iloc[0]['model_name']
        real_model_name = RENAMES.get(extract_model_name(model_name), model_name)

        print(f">> model name '{real_model_name}'")

        scored_rows = []
        for category, rows in data.groupby('category'):
            print(f">> processing category '{category}'")
            new_rows = rows.to_dict('records')

            model_name     =  new_rows[0]['model_name']
            real_model_name = RENAMES.get(extract_model_name(model_name), model_name)
            
            assert len(new_rows) == 50
            scores = score_samples(
                judge,
                [ 
                    SamplePair(
                        prompt   = row['prompt'].split('</think>')[-1].strip(),
                        response = row['model_response'] 
                    ) 
                    for row in new_rows 
                ],
                cast(str, category),
                max_connections=50
            )

            scored_rows.extend(
                row | asdict(score) | { 'model_name': real_model_name } for row, score in zip(new_rows, scores)
            )
            
        frame = pd.DataFrame(data = scored_rows)
        frame.set_index('prompt_id', inplace=True)
        print(f'>> writing results to {out_file}')
        frame.to_csv(out_file)


def score_models(models_dir : str, judge : Model):
    log_files = glob(f'{models_dir}/*.csv')

    if len(log_files) == 0:
        legacy_score_models(models_dir, judge)
    else:
        new_score_models(log_files, judge)

def format_path(*path_to_file : str) -> str: 
    """ Preffixes file name with a timestamp and replace all '/' to '-' in the file portion of of the path"""
    time_portion = time.strftime("%Y-%m-%dT%H-%M-%S%z")
    file_name = f'{time_portion}_{path_to_file[-1].replace('/', '-')}' 
    return file_name if len(path_to_file) == 1 else os.path.join(*path_to_file[:-1], file_name)

@dataclass
class ScoreConfig:
    base_path : str | list[str]

@dataclass
class GenerationConfig:
    model_name_or_path : str
    output_base_path : str | None = None
    system_prompt : str | None    = None

def generate_responses(config : GenerationConfig):
    from vllm import SamplingParams
    model = HuggingFaceModel(
        config.model_name_or_path,
        system_prompt   = config.system_prompt,
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
        )
    )

    data  = pd.read_csv('./resources/prompts.csv')
    assert len(data) == 400
    prompts = list(data['prompt'])
    result = model.generate_with_debug(
        prompts, max_connections = len(prompts),
    )
    frame = pd.DataFrame(
        data = list(zip(
            data['id'],               # prompt_id
            data['category'],         # category
            [config.model_name_or_path] * len(prompts),   # model
            prompts,                  # prompt
            result.outputs,           # model_response
            result.templated_inputs   # raw_model_input
        )),
        columns = ['prompt_id', 'category', 'model_name', 'prompt', 'model_response', 'raw_model_input'] # pyright: ignore
    )

    frame.set_index('prompt_id', inplace=True)
    output_file = format_path(
        config.output_base_path or '.',
        config.model_name_or_path   \
                .strip('/')         \
                .lower()            \
                .replace('_', '-')  \
                .replace('/', '_') + '.csv'
    )

    frame.to_csv(output_file)
    print(f"saved results in '{output_file}'")

# TODO: https://ai.google.dev/gemini-api/docs/batch-api?batch=file (look at this)
def main(): 
    args   = sys.argv[1:] if len(sys.argv) > 1 else ['--help']
    choice = tyro.cli(
        Union[
            Annotated[GenerationConfig, tyro.conf.subcommand(name="generate")],
            Annotated[ScoreConfig,   tyro.conf.subcommand(name="score")],
        ], args=args # pyright: ignore
    ) # pyright: ignore

    if isinstance(choice, GenerationConfig):
        generate_responses(choice)
    else:
        assert isinstance(choice, ScoreConfig)
        load_dotenv()
        judge = Gemini('gemini-2.5-pro')
        paths = [choice.base_path] if isinstance(choice.base_path, str) else choice.base_path
        for path in paths:
            score_models(path, judge)

if __name__ == '__main__': main()
