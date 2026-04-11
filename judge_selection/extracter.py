
import re
from prompts import PromptType

def extract_response(prompt_type: PromptType, response: str) -> dict:
    response = response.replace("**", "")
    # if there's any <reasoning>...</reasoning>, remove it
    if "<reasoning>" in response and "</reasoning>" in response:
        response = response.split("</reasoning>")[-1]
    if prompt_type == PromptType.PROMPT_3_SCORES:
        return extract_response_3_scores(response)
    elif prompt_type == PromptType.PROMPT_1_SCORE:
        return extract_response_1_score(response)
    elif prompt_type == PromptType.PROMPT_3_SCORES_PT:
        return extract_response_3_scores_pt(response)
    elif prompt_type == PromptType.PROMPT_1_SCORE_PT:
        return extract_response_1_score_pt(response)
    else:
        raise ValueError("Invalid prompt type")

def extract_response_3_scores(response: str) -> dict:
    # Use re to get Reasoning and Score from Quality, Accuracy and Completeness
    # Quality
    quality_reasoning = re.search(r"Quality Reasoning: (.+)", response)
    quality_score = re.search(r"Quality Score: (\d+)", response)

    # Accuracy
    accuracy_reasoning = re.search(r"Accuracy Reasoning: (.+)", response)
    accuracy_score = re.search(r"Accuracy Score: (\d+)", response)

    # Completeness
    completeness_reasoning = re.search(r"Completeness Reasoning: (.+)", response)
    completeness_score = re.search(r"Completeness Score: (\d+)", response)

    overall_score = (int(quality_score.group(1)) + int(accuracy_score.group(1)) + int(completeness_score.group(1))) / 3
    try:
        result = {
            "quality_reasoning": quality_reasoning.group(1),
            "quality_score": quality_score.group(1),
            "accuracy_reasoning": accuracy_reasoning.group(1),
            "accuracy_score": accuracy_score.group(1),
            "completeness_reasoning": completeness_reasoning.group(1),
            "completeness_score": completeness_score.group(1),
            "overall_score": overall_score
        }
    except:
        print("\n\n\n")
        print(response)
        print("\n\n\n")
        raise Exception("Error extracting response")
    return result

def extract_response_1_score(response: str) -> dict:
    reasoning = re.search(r"Reasoning: (.+)", response)
    score = re.search(r"Overall Score: (\d+)", response)
    try:
        result = {
            "reasoning": reasoning.group(1),
            "overall_score": float(score.group(1))
        }
    except:
        print("\n\n\n")
        print(response)
        print("\n\n\n")
        raise Exception("Error extracting response")
    return result

def extract_response_3_scores_pt(response: str) -> dict:
    quality_reasoning = re.search(r"Raciocínio Qualidade: (.+)", response)
    quality_score = re.search(r"Pontuação Qualidade: (\d+)", response)

    accuracy_reasoning = re.search(r"Raciocínio Precisão: (.+)", response)
    accuracy_score = re.search(r"Pontuação Precisão: (\d+)", response)

    completeness_reasoning = re.search(r"Raciocínio Completude: (.+)", response)
    completeness_score = re.search(r"Pontuação Completude: (\d+)", response)

    overall_score = (int(quality_score.group(1)) + int(accuracy_score.group(1)) + int(completeness_score.group(1))) / 3
    try:
        result = {
            "quality_reasoning": quality_reasoning.group(1),
            "quality_score": quality_score.group(1),
            "accuracy_reasoning": accuracy_reasoning.group(1),
            "accuracy_score": accuracy_score.group(1),
            "completeness_reasoning": completeness_reasoning.group(1),
            "completeness_score": completeness_score.group(1),
            "overall_score": overall_score
        }
    except:
        print("\n\n\n")
        print(response)
        print("\n\n\n")
        raise Exception("Error extracting response")
    return result

def extract_response_1_score_pt(response: str) -> dict:
    reasoning = re.search(r"Raciocínio:\s*(.+?)\s*Pontuação Global:", response, re.DOTALL)
    score = re.search(r"Pontuação Global:\s*(\d+)", response)
    try:
        result = {
            "reasoning": reasoning.group(1).strip(),
            "overall_score": float(score.group(1))
        }
    except:
        print("\n\n\n")
        print(response)
        print("\n\n\n")
        raise Exception("Error extracting response")
    return result

if __name__ == "__main__":
    text = """
Raciocínio: Claro vamos ver
1.  Precisão: A definição de palavras homógrafas está fundamentalmente incorreta. A resposta afirma que as palavras homógrafas possuem "o mesmo significado", o que é falso; homógrafas têm a mesma grafia mas significados diferentes (e podem ou não ter pronúncias diferentes). O exemplo fornecido ("corda" vs "corda") não ilustra adequadamente o conceito de homógrafas com pronúncias diferentes ou significados distintos, sendo mais duas utilizações da mesma palavra com o mesmo som e significado muito próximo.
2.  Qualidade: A resposta contém um erro ortográfico ("sgnificado" em vez de "significado") e utiliza a grafia brasileira para "homógrafas" ("homôgrafas"), o que vai contra a instrução de Português Europeu. A construção "sendo que" também é menos formal.
3.  Completude: Embora a resposta tente dar uma definição e um exemplo, a imprecisão e incorreção do conteúdo tornam a resposta inútil e, portanto, muito incompleta no que diz respeito a fornecer uma informação correta e relevante. O exemplo não é pertinente para o conceito.

Pontuação Global: 1
    """
    result = extract_response_1_score_pt(text)
    print(result)