from dataclasses import dataclass
from google.genai.types import GenerateContentConfig
from openai import OpenAI
from google import genai 
from itertools import batched
from dotenv import load_dotenv
from google.genai.types import HttpOptions
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, cast
from tqdm import tqdm

import os, time, sys

MAX_RETRIES = 10

def parallel_generation(
        func : Callable[[str], str],
        prompts : list[str], 
        max_connections: int
    ) -> list[str]:

    def wrapper(prompt : str) -> str:
        res = func(prompt)
        assert res, f"response of '{prompt[-80:]}' is None"
        return res


    def retry(prompt : str) -> str:
        for i in range(MAX_RETRIES):
            try: return wrapper(prompt)
            except Exception as ex:
                prompt_msg = prompt if len(prompt) <= 50 else (prompt[:50] + '...')
                times = i + 1
                print("Error doing API call :", ex, file=sys.stderr)
                print(f"RETRY {times:02}/{MAX_RETRIES} '{prompt_msg.strip()}'", file=sys.stderr)
                time.sleep(1.5 * times)

        return wrapper(prompt)

    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        return [result for result in tqdm(executor.map(retry, prompts), total=len(prompts))]

class Model:
    def get_name(self) -> str: ...
    def generate_in_batch(self, prompts: list[str], max_connections : int) -> list[str]: ...
    def generate(self, prompts: list[str]) -> list[str]: 
        return self.generate_in_batch(prompts, 10)

class ChatGPT(Model):
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def get_name(self) -> str:
        return self.model_name

   
    def generate_in_batch(self, prompts: list[str], max_connections : int) -> list[str]:
        return parallel_generation(lambda prompt: 
            self.client.chat.completions.create(
                model=self.model_name, messages=[{'role': 'user', 'content': prompt }]
            ).choices[0].message.content, # pyright: ignore
            prompts, max_connections
        ) 

class Gemini(Model):
    # def __init__(self, model_name : str, **args):
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.client     = genai.Client(
            api_key=os.getenv('GEMINI_API_KEY'),
            # http_options=HttpOptions(timeout=30000),
        )

    def get_name(self) -> str:
        return self.model_name

    def generate_in_batch(self, prompts: list[str], max_connections : int) -> list[str]:
        return parallel_generation(lambda prompt: 
            self.client.models.generate_content(
                model=self.model_name, contents=prompt,
                config=GenerateContentConfig(temperature=0) # in Iago's words: to make it more stable
            ).text, # pyright: ignore
            prompts, max_connections
        ) 

@dataclass
class GenerationResult:
    outputs : list[str]
    templated_inputs : list[str]
    
class HuggingFaceModel(Model):
    def __init__(self, model_name : str, system_prompt : str | None = None, **gen_args):
        from vllm import LLM
        # self.max_connections = max_connections
        self.model_name = model_name
        self.model = LLM(model=model_name)
        self.extra_msgs = [{"role": "system", "content": system_prompt }] if system_prompt else []
        self.get_args = gen_args

    def get_name(self) -> str:
        return self.model_name

    def generate_with_debug(self, prompts: list[str], max_connections : int) -> GenerationResult:
        processed_prompts = cast(list[str], [
            self.model.get_tokenizer().apply_chat_template(
                self.extra_msgs + [{"role": "user", "content": prompt }],  # pyright: ignore
                  tokenize=False, add_generation_prompt=True
            ) for prompt in prompts
        ])

        results = []
        for batch in batched(processed_prompts, max_connections):
            results.extend(self.model.generate(
                batch, **self.get_args 
            ))

        return GenerationResult(
            outputs = [v.outputs[0].text for v in results],
            templated_inputs = processed_prompts
        )

    def generate_in_batch(self, prompts: list[str], max_connections : int) -> list[str]:
        return self.generate_with_debug(prompts, max_connections).outputs

