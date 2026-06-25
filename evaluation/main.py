#!/usr/bin/env python3

from dataclasses import asdict, dataclass, field
import os, sys, time, tyro, torch
import pandas as pd

from typing import Union, Annotated
from dotenv import load_dotenv
from glob import glob

from models import Gemini, ChatGPT, HuggingFaceModel, Model
from scorer import score_samples, SamplePair, ScoreExtractionException, MAX_JUDGE_RESPONSE_TOKENS
from typing import cast

from datasets import load_dataset

RENAMES = {
    # DEFINE ANY RENAME YOU WANT TO IN THE FORM (only for convenience in the final result):
    "Qwen2.5-7B-Instruct": "Qwen 2.5-7B",
	"58_continue_57_carminho-9B-32k-lr_1e-4_to_1e-5_fixed-from_base/checkpoint-154" : "57-58-sft", 
    "57_58-9b-mix-dpo-fixed_ge0_01/checkpoint-3280"    : "57-58-dpo",
	"50-dpo_RLVR-IF-Math-Stem-cascade/checkpoint_400"  : "50-rlvr",
	"57-carminho-22B-32k/checkpoint-5439"              : "57-22b-sft",
	"57-22b-mix-dpo-fixed_ge0_01/checkpoint-3314"      : "57-22b-dpo",
}

@dataclass
class ModelSpec:
    model_name    : str
    alloc_gpu_nr  : int
    max_tokens    : int
    system_prompt : str = ''
    hf_extra_args : dict = field(default_factory=dict)

API_MODELS = {
    'google://': Gemini,
    'openai://': ChatGPT
}

def load_model(spec : ModelSpec) -> Model:
    name_or_path = spec.model_name

    for prefix, builder in API_MODELS.items():
        if name_or_path.startswith(prefix):
            model_id = name_or_path.removeprefix(prefix)
            print(f"loading API model '{model_id}'")
            return builder(model_id)
    
    
    from vllm import SamplingParams
    alloc_gpu_nr = spec.alloc_gpu_nr if spec.alloc_gpu_nr else torch.cuda.device_count()
    vllm_cfg     = spec.hf_extra_args.get('vllm_cfg', {})
    vllm_cfg['tensor_parallel_size'] = alloc_gpu_nr
    spec.hf_extra_args['vllm_cfg']   = vllm_cfg

    return HuggingFaceModel(
        spec.model_name,
        system_prompt = spec.system_prompt,
        vllm_gen_cfg  = {
            'sampling_params': SamplingParams(
                temperature=0.0,
                max_tokens=spec.max_tokens,
                seed=42,
            ),
        },

        **spec.hf_extra_args
    )

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
        errorsome_records_nr = len(checkpoint_categories) * 100 - len(checkpoint_data)
        print(f'>> loaded {len(checkpoint_data)} entries from saved checkpoint')
        if errorsome_records_nr > 0:
            print(f'>> left out {errorsome_records_nr} errorsome records')

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

        model_name      = new_rows[0]['model_name']
        real_model_name = RENAMES.get(extract_model_name(model_name), model_name)
        
        assert len(new_rows) == 100

        scores = score_samples(
            judge,
            [ 
                SamplePair(
                    prompt   = row['prompt'],
                    response = str(row['model_response']).split('</think>')[-1].strip(), 
                ) 
                for row in new_rows 
            ],
            model_name,
            cast(str, category),
            max_connections=50
        )

        new_entries = [
            row | asdict(score) | { 'model_name': real_model_name } for row, score in zip(new_rows, scores)
            if not (score.score == 0 and score.explanation == 'ERROR')
        ]

        if len(new_entries) < 100:
            print(f'>> detected {100 - len(new_entries)} errorsome entries')

        scored_rows.extend(new_entries)
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
    '''Scores generated responses with a judge model'''
    base_path : str | list[str] # A list of dirname or filename with the results from the `generate` command.
    judge_model : str = 'google://gemini-2.5-pro' # Use `openai://<model_id>`, `google://<model_id>` or `<path_to_model_or_hf_name>` to used respective an OpenAI model, a Google Gemini model or a local model.
    alloc_gpu_nr : int = 0 # The number of GPUs be used on the judge model. Use 0 to refer to all available GPUs. It's an option reserved for hugging face models only.

@dataclass
class GenerationConfig:
    '''Prompts provided model to generate responses to ALBA'''
    model_name_or_path : str # Use `openai://<model_id>`, `google://<model_id>` or `<path_to_model_or_hf_name>` to used respective an OpenAI model, a Google Gemini model or a local model.
    output_base_path : str | None = None
    system_prompt : str | None    = None
    alloc_gpu_nr : int = 0 # The number of GPUs be used on the judge model. Use 0 to refer to all available GPUs. It's an option reserved for hugging face models only.

@dataclass
class AggregationConfig:
    '''Aggregates scored responses by model and ALBA category in a single CSV file'''
    base_path : str
    results_file : str = 'alba_results.csv'

def generate_responses(config : GenerationConfig):

    try:
        data  = load_dataset('carminho/alba', 'prompts')['train'].to_pandas()
    except Exception as e:
        print(e)
        print("problem loading the dataset from HuggingFace, falling back to local CSV file")
        csv_path = os.path.join(os.path.dirname(__file__), 'alba_prompts.csv')
        data = load_dataset("csv", data_files=csv_path)['train'].to_pandas()

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 800

    prompts = list(data['prompt'])
    model   = load_model(ModelSpec(
        model_name   = config.model_name_or_path,
        alloc_gpu_nr = config.alloc_gpu_nr,
        max_tokens   = 1024,
        hf_extra_args={
            # if you fall into MEMORY problems, please touch the configs below :)
            'vllm_cfg': {
                    'gpu_memory_utilization'   : 0.80,
                    'disable_custom_all_reduce': True,
                    # 'dtype': 'bfloat16',
                }
            }
    ))

    result = model.generate_with_debug(
        prompts, max_connections = 50,
    )

    frame = pd.DataFrame(
        data = list(zip(
            data['prompt_id'],        # prompt_id
            data['category'],         # category
            [config.model_name_or_path] * len(prompts), # model
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
        judge = load_model(
            ModelSpec(
                model_name=choice.judge_model,
                alloc_gpu_nr=choice.alloc_gpu_nr,
                max_tokens=MAX_JUDGE_RESPONSE_TOKENS,
                hf_extra_args={
                    'vllm_cfg': {
                        'gpu_memory_utilization'   : 0.80,
                        'disable_custom_all_reduce': True,
                        'dtype': 'bfloat16',
                    }
                }
            )
        )
        paths = [choice.base_path] if isinstance(choice.base_path, str) else choice.base_path
        for path in paths:
            score_models(path, judge)
    else:
        assert isinstance(choice, AggregationConfig)
        aggregate_results(choice)

if __name__ == '__main__':
    main()
