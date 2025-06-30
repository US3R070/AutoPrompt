import os

DEFAULT_REASONER_PROMPT = (
    "你是一個專業的 prompt 工程師。以下是舊的 prompt 及其導致的錯誤案例和成功率，"
    "請分析這個 prompt 為什麼會導致這些錯誤，\n\n"
    "舊 prompt:\n{old_prompt}\n\n"
    "錯誤案例:\n{error_cases}\n"
    "目前成功率: {success_rate}\n\n"
    "請用大約70字給出並給出這段 prompt 需要改進的地方以及程度(微調、局部重寫、整體重寫)"
)

class Reasoner:
    def __init__(self, llm, analysis_prompt_template=None):
        self.llm = llm  # LLM 實例
        self.analysis_prompt_template = analysis_prompt_template or DEFAULT_REASONER_PROMPT

    def analyze(self, old_prompt, error_samples, success_rate=None):
        """
        old_prompt: str，舊的 prompt
        error_samples: List[Dict]，每個 dict 包含 input, prediction, label
        success_rate: float，目前正確率
        return: str，錯誤分析說明
        """
        error_cases_str = "\n".join([
            f"輸入: {e['input']}\n預測: {e['prediction']}\n正確: {e['label']}" for e in error_samples
        ])
        analysis_prompt = self.analysis_prompt_template.format(
            old_prompt=old_prompt,
            error_cases=error_cases_str,
            success_rate=(success_rate if success_rate is not None else 'N/A')
        )
        # 調用 LLM 進行分析
        result = self.llm.invoke(analysis_prompt)
        # 若回傳為 dict 且有 'text'，取 text
        if isinstance(result, dict) and 'text' in result:
            analysis = result['text']
        else:
            analysis = str(result)
        return analysis 