from dataclasses import dataclass
from models import Model
import pandas as pd
import re, sys

PROMPT_1_SCORE_PT = """
És um Classificador Profissional de Texto em Português Europeu.
Vais avaliar uma determinada interação na categoria de: {category}

Terás de pontuar de 1 a 5, aqui estão as instruções:
{score_instructions}

Para te ajudar na tua avaliação, aqui estão alguns exemplos:
{few_shots}

A tua resposta deve estar no seguinte formato:
{format_response}

Agora é a tua vez, analisa esta interação e responde:
Prompt: {prompt}
Resposta: {answer}
"""

INSTRUCTIONS_1_SCORE_PT = """
1 - Muito Mau
    Muito Imprecisa	Quase nada está correto ou preciso, se é que algo está.
    Qualidade Muito Baixa	A resposta carece de formatação, a linguagem usada é confusa e difícil de seguir, e/ou a qualidade da linguagem é muito baixa (ocorrência de PT-BR, outras línguas ou erros ortográficos).
    Muito Incompleta	Quase não há contexto e/ou justificação dada com a resposta ou é completamente irrelevante/mal justificada.
2 - Mau	
    Imprecisa	Há várias imprecisões ao longo da resposta.
    Qualidade Baixa	A resposta não está bem estruturada, a linguagem carece de clareza e/ou tem vários erros (PT-BR, erros ortográficos, etc).
    Incompleta	Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
3 - Algo Mau	
    Algo Imprecisa	Há alguma informação imprecisa ou incorreta.
    Qualidade Aceitável	A resposta está razoavelmente bem formatada e clara, e tem apenas alguns erros.
    Algo Incompleta	Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
4 - Algo Bom	
    Algo Precisa	Quase não há imprecisões ou erros.
    Boa Qualidade	A resposta está maioritariamente bem estruturada e clara. Quase não há erros.
    Algo Completa	A resposta fornece algum contexto/justificação/explicação que tem relevância.
5 - Muito Bom	
    Muito Precisa	Não há imprecisões ou erros.
    Qualidade Muito Alta	A resposta está muito bem estruturada, a linguagem é clara e fácil de entender. A resposta não tem erros gramaticais ou ortográficos nem ocorrências de PT-BR.
    Muito Completa	A resposta está completa, bem justificada e fornece contexto adicional.
"""

FORMAT_RESPONSE_1_SCORE_PT = """
Raciocínio: <raciocínio>
Pontuação Global: <pontuação de 1 a 5>
"""

# TODO: should I just add this inline?
FEWSHOT_CSV = pd.read_csv("resources/fewshot_samples.csv", index_col='id')#, index_col=)

@dataclass
class SamplePair:
    prompt: str
    response: str

@dataclass
class Metric:
    score: int
    explanation: str
    judge_prompt : str
    judge_plain_answer: str

def format_prompt_1_score_pt(prompt: str, answer: str, few_shots: list, category: str) -> str:
    few_shots_str = ""
    for few_shot in few_shots:
        few_shots_str += f"""
        Prompt: {few_shot["prompt"]}
        Resposta: {few_shot["answer-1"]}
        Pontuação Global: {few_shot["rating-1"]}

        Prompt: {few_shot["prompt"]}
        Resposta: {few_shot["answer-2"]}
        Pontuação Global: {few_shot["rating-2"]}

        Prompt: {few_shot["prompt"]}
        Resposta: {few_shot["answer-3"]}
        Pontuação Global: {few_shot["rating-3"]}
\n\n
    """
    
    return PROMPT_1_SCORE_PT.format(
        category=category,
        score_instructions=INSTRUCTIONS_1_SCORE_PT,
        few_shots=few_shots_str,
        format_response=FORMAT_RESPONSE_1_SCORE_PT,
        prompt=prompt,
        answer=answer
    )

def extract_response_1_score_pt(judge_prompt : str, response: str) -> Metric:
    reasoning = re.search(r"Raciocínio:\s*(.+?)\s*Pontuação Global:", response, re.DOTALL)
    score = re.search(r"Pontuação Global:\s*(\d+)", response)
    try:
        return Metric(
            judge_prompt= judge_prompt,
            judge_plain_answer= response,
            explanation = reasoning.group(1).strip(), # pyright: ignore
            score = float(score.group(1)) # pyright: ignore
        )
    except:
        print("\n\n\n", file=sys.stderr)
        print(response, file=sys.stderr)
        print("\n\n\n", file=sys.stderr)
        raise Exception("Error extracting response")

def score_samples(
        judge : Model,
        pairs : list[SamplePair],
        category : str,
        max_connections = 20
    ) -> list[Metric]:

    category_shots = FEWSHOT_CSV[FEWSHOT_CSV['category'] == category]
    assert len(category_shots) > 0
    assert all(map(lambda pair: len(pair.response) > 0, pairs))

    few_shot_examples = [
        { 
            "id": id,
            "prompt": row["Prompt"],            # pyright: ignore
            "answer-1": row["Answer A (5)"],    # pyright: ignore
            "rating-1": row["Rating"],          # pyright: ignore
            "answer-2": row["Answer B (4-3)"],  # pyright: ignore
            "rating-2": row["Rating.1"],        # pyright: ignore
            "answer-3": row["Answer C (2-1)"],  # pyright: ignore
            "rating-3": row["Rating.2"]         # pyright: ignore
        } 
        for id, row in category_shots.iterrows()
    ]

    judge_prompts = [
         format_prompt_1_score_pt(pair.prompt, pair.response, few_shot_examples, category) for pair in pairs
    ]

    return [
        extract_response_1_score_pt(judge_prompt, response) for judge_prompt, response in zip(judge_prompts, judge.generate_in_batch(judge_prompts, max_connections))
    ]


# FEWSHOT_CSV['category'] = FEWSHOT_CSV['category'].replace('Culture-Bound Semantics', 'Culture-bound Semantics') # TODO: do the inverse of this everywhere
# print(FEWSHOT_CSV['category'].unique())
