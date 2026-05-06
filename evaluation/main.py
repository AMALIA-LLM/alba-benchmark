#!/usr/bin/env python3

from dataclasses import asdict, dataclass
import os, sys, time, tyro
import pandas as pd

from typing import Union, Annotated
from dotenv import load_dotenv
from glob import glob

from models import Gemini, ChatGPT, HuggingFaceModel, Model
from scorer import score_samples, SamplePair, ScoreExtractionException, MAX_JUDGE_RESPONSE_TOKENS
from typing import cast, Literal

from datasets import load_dataset

RENAMES = {
    "47-32k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-2875": "AMALIA-9B 32k v49",
    "47-4k-9B-carminho-with_euroblocks_safety_hermes_customst/checkpoint-13590": "AMALIA-9B 4k v49 (checkpoint 13590)",
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

def extract_model_name(model_name : str) -> str:
    parts = model_name.split('/')
    if not parts[-1]: parts.pop()
    return '/'.join(parts[-2:]) if parts[-1].startswith('checkpoint') else parts[-1]

def save_checkpoint(entries : list[dict], filename : str):
    frame = pd.DataFrame(data = entries)
    frame.set_index('prompt_id', inplace=True)
    frame.to_csv(filename)

def score_model_file(model_file : str, judge : Model):
    if model_file.endswith('_scored.csv'): return

    print(f"> loading '{model_file}'")
    out_file = f'{model_file.rstrip('.csv')}_scored.csv'

    if os.path.isfile(out_file):
        print('>> skipping probably already scored ...')
        return

    checkpoint_file = os.path.join(
        os.path.dirname(out_file),
        f'.{os.path.basename(out_file)}.checkpoint'
    )

    checkpoint_data = []
    checkpoint_categories = set()
    if os.path.isfile(checkpoint_file):
        checkpoint = pd.read_csv(checkpoint_file)
        checkpoint_categories.update(
            checkpoint['category'].unique()
        )
        checkpoint_data.extend( checkpoint.to_dict('records') )
        assert len(checkpoint_data) == len(checkpoint_categories) * 100

        print(f'>> loaded {len(checkpoint_data)} entries from saved checkpoint')

    data = pd.read_csv(model_file)
    model_name      =  data.iloc[0]['model_name']
    real_model_name = RENAMES.get(extract_model_name(model_name), model_name)

    print(f">> model name '{real_model_name}'")
    scored_rows = [] + checkpoint_data
    for category, rows in data.groupby('category'):
        if category in checkpoint_categories:
            print(f">> category '{category}' found in checkpoint, skipping it ...")
            continue

        print(f">> processing category '{category}'")
        new_rows = rows.to_dict('records')

        model_name     =  new_rows[0]['model_name']
        real_model_name = RENAMES.get(extract_model_name(model_name), model_name)
        
        assert len(new_rows) == 100
        scores = score_samples(
            judge,
            [ 
                SamplePair(
                    prompt   = row['prompt'],
                    response = row['model_response'].split('</think>')[-1].strip(), 
                ) 
                for row in new_rows 
            ],
            cast(str, category),
            max_connections=50
        )

        scored_rows.extend(
            row | asdict(score) | { 'model_name': real_model_name } for row, score in zip(new_rows, scores)
        )

        save_checkpoint(scored_rows, checkpoint_file)

    os.rename(checkpoint_file, out_file)
    print(f'>> saving results as {out_file}')

def score_with_retries(model_file : str, judge : Model, max_retry = 2):
    for i in range(max_retry + 1):
        try: return score_model_file(model_file, judge)
        except ScoreExtractionException:
            print(f'{i + 1}/{max_retry} retrying to score \'{extract_model_name(model_file)}\'')

def score_models(models_dir : str, judge : Model):
    models_list   = glob(f'{models_dir}/*.csv') if os.path.isdir(models_dir) else [models_dir]
    failed_models = []

    for model_file in models_list:
        try:
            score_with_retries(model_file, judge)
        except Exception as e:
            print(f'* failed to score \'{model_file}\':', e, file=sys.stderr)
            failed_models.append(model_file)

    if len(failed_models) > 0:
        print('** failed to models:', *failed_models, sep='\n- ',file=sys.stderr)
        sys.exit(1)

def format_path(*path_to_file : str) -> str: 
    """ Preffixes file name with a timestamp and replace all '/' to '-' in the file portion of of the path"""
    time_portion = time.strftime("%Y-%m-%dT%H-%M-%S%z")
    file_name = f'{time_portion}_{path_to_file[-1].replace('/', '-')}' 
    return file_name if len(path_to_file) == 1 else os.path.join(*path_to_file[:-1], file_name)

@dataclass
class ScoreConfig:
    base_path : str | list[str] # a list of dirname or filename with the results from the `generate` command
    judge : Literal['gpt-oss', 'gemini'] = 'gemini'

@dataclass
class GenerationConfig:
    model_name_or_path : str
    output_base_path : str | None = None
    system_prompt : str | None    = None

@dataclass
class AggregationConfig:
    base_path : str
    results_file : str = 'alba_results.csv'

def load_model(config : GenerationConfig) -> Model:
    name_or_path = config.model_name_or_path
    api_models = [
        ('gemini/', Gemini),
        ('chatgpt/', ChatGPT)
    ]

    for prefix, builder in api_models:
        if name_or_path.startswith(prefix):
            model_id = '/'.join( name_or_path.split('/')[1:] )
            print(f"loading API model '{model_id}'")
            return builder(model_id)
    
    from vllm import SamplingParams
    return HuggingFaceModel(
        config.model_name_or_path,
        system_prompt = config.system_prompt,
        generation_cfg = {
            'sampling_params': SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                seed=42,
            )
        }
    )

def generate_responses(config : GenerationConfig):
    model = load_model(config)
    data  = load_dataset('carminho/alba', 'prompts')['train'].to_pandas()

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 800

    prompts = list(data['prompt'])
    result  = model.generate_with_debug(
        prompts, max_connections = 50,
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


def get_judge_model(judge : str) -> Model:
    if judge == 'gemini':
        return Gemini('gemini-2.5-pro')

    if judge == 'gpt-oss':
        from vllm import SamplingParams
        return HuggingFaceModel(
            model_name = 'openai/gpt-oss-120b',
            generation_cfg   = { 
                'sampling_params': SamplingParams(temperature=0.0, max_tokens=MAX_JUDGE_RESPONSE_TOKENS, seed=42) 
            },
            llm_cfg = {
                'gpu_memory_utilization'   : 0.80,
                'disable_custom_all_reduce': True,
                'dtype': 'bfloat16',
            },
    )

    raise ValueError(f"invalid judge '{judge}' expected either 'gemini' or 'gpt-oss'")

def aggregate_results(cfg : AggregationConfig):

    results_pattern = os.path.join(cfg.base_path, '*_scored.csv')
    results_file_list = glob(results_pattern) if os.path.isdir(cfg.base_path) else [cfg.base_path]

    entries = []
    for file in results_file_list:
        print(f'> processing \'{file}\'')
        data  = pd.read_csv(file)
        model = data.iloc[0]['model_name']

        results = cast(pd.Series, data.groupby('category')['score'].mean())
        entries.append(
            { 'model': model } | { key : round(((value - 1) / 4) * 100, 2) for key, value in results.items()}
        )

    frame = pd.DataFrame(data = entries)
    frame.set_index('model', inplace=True)
    frame.to_csv(cfg.results_file)
    print(f'saving results to \'{cfg.results_file}\'')

def main(): 
    args   = sys.argv[1:] if len(sys.argv) > 1 else ['--help']
    choice = tyro.cli(
        Union[
            Annotated[GenerationConfig, tyro.conf.subcommand(name="generate")],
            Annotated[ScoreConfig,   tyro.conf.subcommand(name="score")],
            Annotated[AggregationConfig,   tyro.conf.subcommand(name="aggregate")],
        ], args=args # pyright: ignore
    ) # pyright: ignore

    load_dotenv()
    if isinstance(choice, GenerationConfig):
        generate_responses(choice)
    elif isinstance(choice, ScoreConfig):
        judge = get_judge_model(choice.judge)
        paths = [choice.base_path] if isinstance(choice.base_path, str) else choice.base_path
        for path in paths:
            score_models(path, judge)
    else:
        assert isinstance(choice, AggregationConfig)
        aggregate_results(choice)

if __name__ == '__main__': main()
