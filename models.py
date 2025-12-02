from google.genai.types import GenerateContentConfig
from openai import OpenAI
from google import genai 
from itertools import batched
from dotenv import load_dotenv
# from vllm import LLM
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
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
    def __init__(self, model_name : str):
        self.model_name = model_name
        self.client = genai.Client(
            api_key=os.getenv('GEMINI_API_KEY')
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
    
# class HuggingFaceModel(Model):
#     def __init__(self, model_name : str):
#         # self.max_connections = max_connections
#         self.model_name = model_name
#         self.model = LLM(model=model_name)
#
#     def get_name(self) -> str:
#         return self.model_name
#
#     def generate_in_batch(self, prompts: list[str], max_connections : int) -> list[str]:
#         results = []
#         for batch in batched(prompts, max_connections):
#             results.extend(self.model.generate(batch))
#         return [v.outputs[0].text for v in results]

