#!/usr/bin/env python3

import sys
import json
import numpy as np
from typing import Any
from dotenv import load_dotenv
from glob import glob
from models import Gemini
from scorer import Metric, score_samples, SamplePair

def extract_bench_id(data : dict[str, Any]) -> str:
    keys = list(data['results'].keys())
    assert len(keys) == 1
    return keys[0]

def get_bench_samples_filename(bench_id : str, model_dir : str) -> str:
    sample_files = glob(f'{model_dir}/samples_{bench_id}*.jsonl')
    assert len(sample_files) == 1, f"found more than 1 samples file for '{bench_id}': {','.join(sample_files)}"
    return sample_files[0]

def rescore_sample(sample : dict, score : Metric) -> dict:
    new_sample = sample.copy() | {
        'metrics': ['score', 'explanation'],
        'score': score.score,
        'explanation': score.explanation
    }

    new_sample.pop('bypass')
    return new_sample

def fix_log_data(log_data : dict[str, Any], bench_id : str, scores : list[Metric]):
    mean, stderr = aggregate_scores(scores)

    # TODO: think if you want to copy this or not
    log_data['results'][bench_id] |= {
      "score,none": mean,
      "score_stderr,none": stderr,
      "explanation,none": 0,
      "explanation_stderr,none": "N/A"
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
        }
    ]

    log_data['higher_is_better'] = {
        bench_id: {
          "score": False,
          "explanation": False
        }
    }
    

def aggregate_scores(scores : list[Metric]) -> tuple[tuple, tuple]:
    plain_scores = [score.score for score in scores]
    return np.mean(plain_scores), np.std(plain_scores) # pyright: ignore


def main(): 
    if len(sys.argv) < 2:
        print(f"usage: python3 {sys.argv[0]} <samples_dir>", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    samples_dir = sys.argv[1]
    print(f"> processing models samples in '{samples_dir}'")

    judge = Gemini('gemini-2.5-pro')
    for model_dir in glob(f'{samples_dir}/*'):
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
            assert all(map(lambda entry: len(entry['filtered_resps']) == 1, samples))
            assert all(map(lambda entry: entry['resps'][0] == entry['filtered_resps'], samples))

            print('>>> generating scores')
            scores = score_samples(
                judge, 
                [SamplePair(prompt = sample['doc']['prompt'], response = sample['resps'][0]) for sample in samples], # TODO: should I do strip?
                # [SamplePair(prompt = sample['doc']['prompt'], response = sample['resps'][0]) for sample in samples],
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
                json.dump(log_data, file, indent=4)

            with open(samples_filename, 'w') as file:
                file.writelines([
                    json.dumps(sample) + '\n' for sample in new_samples
                ])

if __name__ == '__main__': main()
