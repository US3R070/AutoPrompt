import os

class Reasoner:
    def __init__(self, llm, analysis_prompt_template):
        self.llm = llm  # LLM 實例
        self.analysis_prompt_template = analysis_prompt_template

    def analyze(self, old_prompt, error_samples):
        """
        old_prompt: str，舊的 prompt
        error_samples: List[Dict]，每個 dict 包含 input, prediction, label
        return: str，錯誤分析說明
        """
        error_cases_str = "\n".join([
            f"輸入: {e['input']}\n預測: {e['prediction']}\n正確: {e['label']}" for e in error_samples
        ])
        analysis_prompt = self.analysis_prompt_template.format(
            old_prompt=old_prompt,
            error_cases=error_cases_str
        )
        # 調用 LLM 進行分析
        result = self.llm.invoke(analysis_prompt)
        # 若回傳為 dict 且有 'text'，取 text
        if isinstance(result, dict) and 'text' in result:
            analysis = result['text']
        else:
            analysis = str(result)
        return analysis 