# ============================================================================
# PROMPT TYPES
# ============================================================================

class PromptType:
    PROMPT_3_SCORES = "PROMPT_3_SCORES"
    PROMPT_1_SCORE = "PROMPT_1_SCORE"
    PROMPT_3_SCORES_PT = "PROMPT_3_SCORES_PT"
    PROMPT_1_SCORE_PT = "PROMPT_1_SCORE_PT"


# ============================================================================
# MAIN FORMATTER
# ============================================================================

def format_prompt(prompt_type: PromptType, prompt: str, answer: str, few_shots: list, category: str) -> str:
    if prompt_type == PromptType.PROMPT_3_SCORES:
        return format_prompt_3_score(prompt, answer, few_shots, category)
    elif prompt_type == PromptType.PROMPT_1_SCORE:
        return format_prompt_1_score(prompt, answer, few_shots, category)
    elif prompt_type == PromptType.PROMPT_3_SCORES_PT:
        return format_prompt_3_score_pt(prompt, answer, few_shots, category)
    elif prompt_type == PromptType.PROMPT_1_SCORE_PT:
        return format_prompt_1_score_pt(prompt, answer, few_shots, category)
    else:
        raise ValueError("Invalid prompt type")


# ============================================================================
# 3-SCORE PROMPT (Quality, Accuracy, Completeness)
# ============================================================================

PROMPT_3_SCORES = """
You are a Professional European Portuguese Text Classifier.
You will be evaluating a certain interaction on the category of: {category}

You'll have to score from 1 to 5, on 3 different aspects, here's the instructions:
{score_instructions}

To help you on your evaluation, here's some examples
{few_shots}

Your own response should be in the following format:
{format_response}

Now it's your turn, analize this interaction and answer yourself:
Prompt: {prompt}
Answer: {answer}
"""

QUALITY_INSTRUCTIONS = """
Score the quality of the answer from 1-5:
1 - Very Low Quality: The answer lacks formatting, the language used is confusing and hard to follow, and/or the language output has very low quality (occurance of PT-BR, other languages or misspellings).
2 - Low Quality: The answer is not well structured, the language lacks clarity and/or has several errors (PT-BR, misspellings, etc).
3 - Acceptable Quality: The answer is somewhat well formatted and clear, and only has a few errors.
4 - Good Quality: The answer is mostly well structured and clear. There are almost no errors.
5 - Very High Quality: The answer is very well structured, the language is clear and easy to understand. The answer has no grammatical or spelling errors or any occurances of PT-BR.
"""

ACCURACY_INSTRUCTIONS = """
Score the accuracy of the answer from 1-5:
1 - Very Inaccurate: There is barely anything right or accurate, if at all.
2 - Inaccurate: There are several inaccuracies throughout the output.
3 - Somewhat Inaccurate: There is some inaccurate or incorrect information.
4 - Somewhat Accurate: There are almost no inaccuracies or errors.
5 - Very Accurate: There are no inaccuracies or errors.
"""

COMPLETENESS_INSTRUCTIONS = """
Score the completeness of the answer from 1-5:
1 - Very Incomplete: There is almost no context and/or justification given with the answer or it's completely irrelevant/badly justified.
2 - Incomplete: There is very little context/justification/explanation being provided or it's not very relevant.
3 - Somewhat Incomplete: There is very little context/justification/explanation being provided or it's not very relevant.
4 - Somewhat Complete: The answer provides some context/justification/explanation that bears relevance.
5 - Very Complete: The answer is complete, well justified, and provides added context.
"""

FORMAT_RESPONSE_3_SCORES = """
Quality Reasoning: <reasoning>
Quality Score: <score>

Accuracy Reasoning: <reasoning>
Accuracy Score: <score>

Completeness Reasoning: <reasoning>
Completeness Score: <score>
"""

def format_prompt_3_score(prompt: str, answer: str, few_shots: list, category: str) -> str:
    few_shots_str = ""
    for few_shot in few_shots:
        few_shots_str += f"""
        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-1"]}
        Overall Score: {few_shot["rating-1"]}

        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-2"]}
        Overall Score: {few_shot["rating-2"]}

        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-3"]}
        Overall Score: {few_shot["rating-3"]}
\n\n
    """
    
    score_instructions = "\n".join([QUALITY_INSTRUCTIONS, ACCURACY_INSTRUCTIONS, COMPLETENESS_INSTRUCTIONS])
    
    return PROMPT_3_SCORES.format(
        category=category,
        score_instructions=score_instructions,
        few_shots=few_shots_str,
        format_response=FORMAT_RESPONSE_3_SCORES,
        prompt=prompt,
        answer=answer
    )


# ============================================================================
# 1-SCORE PROMPT (Overall Score)
# ============================================================================

PROMPT_1_SCORE = """
You are a Professional European Portuguese Text Classifier.
You will be evaluating a certain interaction on the category of: {category}

You'll have to score from 1 to 5, here's the instructions:
{score_instructions}

To help you on your evaluation, here's some examples
{few_shots}

Your own response should be in the following format:
{format_response}

Now it's your turn, analize this interaction and answer yourself:
Prompt: {prompt}
Answer: {answer}
"""

INSTRUCTIONS_1_SCORE = """
1 - Very Bad
    Very Inaccurate	There is barely anything right or accurate, if at all.
    Very Low Quality	The answer lacks formatting, the language used is confusing and hard to follow, and/or the language output has very low quality (occurance of PT-BR, other languages or misspellings).
    Very Incomplete	There is almost no context and/or justification given with the answer or it's completely irrelevant/badly justified.
2 - Bad	
    Inaccurate	There are several inaccuracies throughout the output.
    Low Quality	The answer is not well structured, the language lacks clarity and/or has several errors (PT-BR, misspellings, etc).
    Incomplete	There is very little context/justification/explanation being provided or it's not very relevant.
3 - Somewhat Bad	
    Somewhat Inaccurate	There is some inaccurate or incorrect information.
    Acceptable Quality	The answer is somewhat well formatted and clear, and only has a few errors.
    Somewhat Incomplete	There is very little context/justification/explanation being provided or it's not very relevant.
4 - Somewhat Good	
    Somewhat Accurate	There are almost no inaccuracies or errors.
    Good Quality	The answer is mostly well structured and clear. There are almost no errors.
    Somewhat Complete	The answer provides some context/justification/explanation that bears relevance.
5 - Very Good	
    Very Accurate	There are no inaccuracies or errors.
    Very High Quality	The answer is very well structured, the language is clear and easy to understand. The answer has no grammatical or spelling errors or any occurances of PT-BR.
    Very Complete	The answer is complete, well justified, and provides added context.
"""

FORMAT_RESPONSE_1_SCORE = """
Reasoning: <reasoning>
Overall Score: <score from 1 to 5>
"""

def format_prompt_1_score(prompt: str, answer: str, few_shots: list, category: str) -> str:
    few_shots_str = ""
    for few_shot in few_shots:
        few_shots_str += f"""
        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-1"]}
        Overall Score: {few_shot["rating-1"]}

        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-2"]}
        Overall Score: {few_shot["rating-2"]}

        Prompt: {few_shot["prompt"]}
        Answer: {few_shot["answer-3"]}
        Overall Score: {few_shot["rating-3"]}
\n\n
    """
    
    return PROMPT_1_SCORE.format(
        category=category,
        score_instructions=INSTRUCTIONS_1_SCORE,
        few_shots=few_shots_str,
        format_response=FORMAT_RESPONSE_1_SCORE,
        prompt=prompt,
        answer=answer
    )


# ============================================================================
# 3-SCORE PROMPT - PORTUGUESE (Qualidade, Precisão, Completude)
# ============================================================================

PROMPT_3_SCORES_PT = """
És um Classificador Profissional de Texto em Português Europeu.
Vais avaliar uma determinada interação na categoria de: {category}

Terás de pontuar de 1 a 5, em 3 aspetos diferentes, aqui estão as instruções:
{score_instructions}

Para te ajudar na tua avaliação, aqui estão alguns exemplos:
{few_shots}

A tua resposta deve estar no seguinte formato:
{format_response}

Agora é a tua vez, analisa esta interação e responde:
Prompt: {prompt}
Resposta: {answer}
"""

QUALITY_INSTRUCTIONS_PT = """
Pontua a qualidade da resposta de 1 a 5:
1 - Qualidade Muito Baixa: A resposta carece de formatação, a linguagem usada é confusa e difícil de seguir, e/ou a qualidade da linguagem é muito baixa (ocorrência de PT-BR, outras línguas ou erros ortográficos).
2 - Qualidade Baixa: A resposta não está bem estruturada, a linguagem carece de clareza e/ou tem vários erros (PT-BR, erros ortográficos, etc).
3 - Qualidade Aceitável: A resposta está razoavelmente bem formatada e clara, e tem apenas alguns erros.
4 - Boa Qualidade: A resposta está maioritariamente bem estruturada e clara. Quase não há erros.
5 - Qualidade Muito Alta: A resposta está muito bem estruturada, a linguagem é clara e fácil de entender. A resposta não tem erros gramaticais ou ortográficos nem ocorrências de PT-BR.
"""

ACCURACY_INSTRUCTIONS_PT = """
Pontua a precisão da resposta de 1 a 5:
1 - Muito Imprecisa: Quase nada está correto ou preciso, se é que algo está.
2 - Imprecisa: Há várias imprecisões ao longo da resposta.
3 - Algo Imprecisa: Há alguma informação imprecisa ou incorreta.
4 - Algo Precisa: Quase não há imprecisões ou erros.
5 - Muito Precisa: Não há imprecisões ou erros.
"""

COMPLETENESS_INSTRUCTIONS_PT = """
Pontua a completude da resposta de 1 a 5:
1 - Muito Incompleta: Quase não há contexto e/ou justificação dada com a resposta ou é completamente irrelevante/mal justificada.
2 - Incompleta: Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
3 - Algo Incompleta: Há muito pouco contexto/justificação/explicação fornecida ou não é muito relevante.
4 - Algo Completa: A resposta fornece algum contexto/justificação/explicação que tem relevância.
5 - Muito Completa: A resposta está completa, bem justificada e fornece contexto adicional.
"""

FORMAT_RESPONSE_3_SCORES_PT = """
Raciocínio Qualidade: <raciocínio>
Pontuação Qualidade: <pontuação>

Raciocínio Precisão: <raciocínio>
Pontuação Precisão: <pontuação>

Raciocínio Completude: <raciocínio>
Pontuação Completude: <pontuação>
"""

def format_prompt_3_score_pt(prompt: str, answer: str, few_shots: list, category: str) -> str:
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
    
    score_instructions = "\n".join([QUALITY_INSTRUCTIONS_PT, ACCURACY_INSTRUCTIONS_PT, COMPLETENESS_INSTRUCTIONS_PT])
    
    return PROMPT_3_SCORES_PT.format(
        category=category,
        score_instructions=score_instructions,
        few_shots=few_shots_str,
        format_response=FORMAT_RESPONSE_3_SCORES_PT,
        prompt=prompt,
        answer=answer
    )


# ============================================================================
# 1-SCORE PROMPT - PORTUGUESE (Pontuação Global)
# ============================================================================

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

if __name__ == "__main__":
    #print(format_prompt_1_score("prompt", "answer", [], "category"))
    print(format_prompt_1_score_pt("<prompt>", "<answer>", [], "<category>"))
    #print(format_prompt_3_score("<prompt>", "<answer>", [], "<category>"))
    #print(format_prompt_3_score_pt("prompt", "answer", [], "category"))
