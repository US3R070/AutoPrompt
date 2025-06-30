# A file containing the json schema for the output of all the LLM chains
# A file containing the parser for the output of all the LLM chains
import re


def prediction_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    # print(response)
    pattern = re.compile(r'Sample (\d+): (\w+)')
    matches = pattern.findall(response['text'])
    predictions = [{'id': int(match[0]), 'prediction': match[1]} for match in matches]
    return {'results': predictions}

def annotation_parser(response: dict) -> dict:
    """
    一個更健壯的、可以容忍 LLM 輸出格式錯誤的解析器。
    """
    text = response.get('text', '')
    # print(response)
    # 更寬鬆的 regex：允許空格，不區分大小寫，處理字號的全形半形
    pattern = re.compile(r'Sample (\d+): (.*?)(?=<eos>|$)', re.DOTALL)
    matches = pattern.findall(text)
    
    if not matches:
        print(f"[Warning] Annotation parser failed to find any matches in text:\n{text}")
        return {'results': []}
    predictions = [{'id': int(match[0]), 'prediction': match[1].strip()} for match in matches]
    return {'results': predictions}

def prediction_generation_parser(response: dict) -> dict:
    """
    Parse the response from the LLM chain
    :param response: The response from the LLM chain
    :return: The parsed response
    """
    pattern = re.compile(r'Sample (\d+): (.*?)(?=<eos>|$)', re.DOTALL)
    matches = pattern.findall(response['text'])
    predictions = [{'id': int(match[0]), 'prediction': match[1].strip()} for match in matches]
    return {'results': predictions}
