# ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs

This repository provides the code and resources for running the **ALBA benchmark**, introduced in the paper  
[ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs](https://arxiv.org/abs/2603.26516), accepted at **PROPOR 2026**.

ALBA is designed to systematically evaluate generative large language models (LLMs) across a range of linguistic and language-specific dimensions in European Portuguese, enabling more rigorous and fine-grained analysis of model performance.

## Code Structure

This repository has **two main pipelines**:

- `evaluation/`: generate model outputs and score model runs.
- `judge_selection/`: benchmark and compare **judge models** (the evaluators themselves) across prompting/few-shot strategies.

## Repository structure

- `evaluation/`
  - `main.py`: CLI entrypoint for response generation and scoring.
  - `models.py`: wrappers for Gemini, ChatGPT, and local Hugging Face/vLLM models.
  - `scorer.py`: judge-scoring logic.
- `judge_selection/`
  - `main.py`: runs judge-selection experiments and aggregates stats.
  - `api.py`: model clients (OpenAI, Gemini, DeepSeek, AWS Bedrock).
  - `choose_few_shot.py`, `prompts.py`, `extracter.py`: few-shot selection, prompt templates, and response parsing.
  - `csvs-train/`, `csvs-test/`: training/test splits used in judge selection.

## What each folder is for

## 1) `evaluation/` (evaluate candidate models)

Use this when you want to:

1. Generate responses from a model on the ALBA prompts dataset.
2. Score those responses with a judge model.

`evaluation/main.py` supports two subcommands:

- `generate`: produces CSV files with model outputs.
- `score`: scores one or more CSV/model-output folders.

### Typical flow

1. Generate outputs from your model.
2. Score those outputs with the judge.
3. Analyze resulting scored CSV files.

## 2) `judge_selection/` (find the best judge configuration)

Use this when you want to decide **which judge model/prompt/few-shot setup** should be used for evaluation.

`judge_selection/main.py`:

- iterates through judge models,
- applies different few-shot strategies and prompt types,
- scores expected ratings from curated CSV splits,
- writes per-category JSON files plus aggregated `Z-Stats.json` summaries.

### Typical flow

1. Run experiments over `csvs-train` / `csvs-test` splits.
2. Compare aggregate stats in result folders.
3. Pick the best judge setup and use it in the evaluation pipeline.

## Requirements

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

## API keys and credentials

These scripts call external APIs. Set credentials before running.

At minimum, configure the keys for the providers you use:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (optional, defaults are used in code)

Both pipelines load environment variables via `python-dotenv`, so a `.env` file in the repo root works well.

Example `.env`:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

## How to run

### `evaluation/`

From repo root:

```bash
python evaluation/main.py generate --model-name-or-path gemini/gemini-2.5-pro --output-base-path outputs
```

Then score:

```bash
python evaluation/main.py score --base-path outputs
```

### `judge_selection/`

Because `judge_selection/main.py` currently uses relative split paths like `csvs-train` / `csvs-test`, run it from inside that folder:

```bash
cd judge_selection
python main.py
```

Results are written to the folders configured in `judge_selection/main.py` (for example `best_judge_results` and `optimal_config_results`).


## How to Cite

If you use this repository or the ALBA benchmark in your work, please cite:

```bibtex
@inproceedings{vieira-etal-2026-alba,
    title = "{ALBA}: A {E}uropean {P}ortuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative {LLM}s",
    author = "Vieira, In{\^e}s  and
      Calvo, In{\^e}s  and
      Paulo, Iago  and
      Furtado, James  and
      Ferreira, Rafael  and
      Tavares, Diogo  and
      Gl{\'o}ria-Silva, Diogo  and
      Semedo, David  and
      Magalh{\~a}es, Jo{\~a}o",
    booktitle = "Proceedings of the 17th International Conference on Computational Processing of {P}ortuguese ({PROPOR} 2026) - Vol. 1",
    month = apr,
    year = "2026",
    address = "Salvador, Brazil",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.propor-1.69/",
    pages = "697--707",
    ISBN = "979-8-89176-387-6"
}
```
