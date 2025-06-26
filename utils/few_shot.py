import pandas as pd
import json
import random
from pathlib import Path
from itertools import combinations
import logging

class FewShotSelector:
    def __init__(self, enable_few_shot=False, num_shots=3, examples_pool=None, output_dir='dump_classification'):
        """
        Few-shot 範例選擇器
        
        Args:
            enable_few_shot: 是否啟用 few-shot
            num_shots: few-shot 的數量
            examples_pool: 可用的範例池 (DataFrame)，包含 text, annotation 欄位
            output_dir: 輸出目錄，用於儲存最佳 few-shot 組合
        """
        self.enable_few_shot = enable_few_shot
        self.num_shots = num_shots
        self.examples_pool = examples_pool if examples_pool is not None else pd.DataFrame()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存每個 prompt 的最佳 few-shot 組合
        self.best_few_shots = {}
        self.few_shot_history = []
        
    def set_examples_pool(self, examples_df):
        """設定可用的範例池"""
        self.examples_pool = examples_df.copy()
        
    def generate_few_shot_text(self, examples):
        """
        將 few-shot examples 轉換成文字格式
        
        Args:
            examples: DataFrame，包含 text, annotation 欄位
            
        Returns:
            str: 格式化的 few-shot 文字
        """
        if examples.empty:
            return ""
            
        few_shot_text = "以下是一些範例：\n\n"
        for idx, row in examples.iterrows():
            few_shot_text += f"範例 {idx + 1}:\n"
            few_shot_text += f"輸入: {row['text']}\n"
            few_shot_text += f"輸出: {row['annotation']}\n\n"
        
        return few_shot_text
    
    def combine_prompt_with_few_shot(self, base_prompt, few_shot_examples):
        """
        將基礎 prompt 與 few-shot examples 結合
        
        Args:
            base_prompt: 基礎 prompt 字串
            few_shot_examples: DataFrame，few-shot 範例
            
        Returns:
            str: 結合後的完整 prompt
        """
        if not self.enable_few_shot or few_shot_examples.empty:
            return base_prompt
            
        few_shot_text = self.generate_few_shot_text(few_shot_examples)
        combined_prompt = f"{few_shot_text}\n現在請根據上述範例進行分類：\n\n{base_prompt}"
        
        return combined_prompt
    
    def sample_few_shot_combinations(self, max_combinations=10):
        """
        從範例池中採樣多種 few-shot 組合
        
        Args:
            max_combinations: 最大組合數量
            
        Returns:
            list: few-shot 組合列表，每個元素是 DataFrame
        """
        if not self.enable_few_shot or len(self.examples_pool) < self.num_shots:
            return [pd.DataFrame()]  # 回傳空的 DataFrame
            
        # 確保每個類別都有代表
        if 'annotation' in self.examples_pool.columns:
            unique_labels = self.examples_pool['annotation'].unique()
            
            # 如果類別數量小於等於 num_shots，每個類別選一個
            if len(unique_labels) <= self.num_shots:
                combinations_list = []
                for _ in range(max_combinations):
                    selected_examples = []
                    for label in unique_labels:
                        label_examples = self.examples_pool[self.examples_pool['annotation'] == label]
                        if len(label_examples) > 0:
                            selected_examples.append(label_examples.sample(1).iloc[0])
                    
                    # 如果還需要更多範例，隨機補充
                    remaining_needed = self.num_shots - len(selected_examples)
                    if remaining_needed > 0:
                        remaining_pool = self.examples_pool[~self.examples_pool.index.isin([ex.name for ex in selected_examples])]
                        if len(remaining_pool) >= remaining_needed:
                            additional = remaining_pool.sample(remaining_needed)
                            selected_examples.extend([additional.iloc[i] for i in range(len(additional))])
                    
                    combination_df = pd.DataFrame(selected_examples[:self.num_shots])
                    combinations_list.append(combination_df)
                
                return combinations_list
        
        # 一般情況：隨機採樣組合
        combinations_list = []
        for _ in range(max_combinations):
            sampled = self.examples_pool.sample(min(self.num_shots, len(self.examples_pool)))
            combinations_list.append(sampled)
            
        return combinations_list
    
    def evaluate_few_shot_combinations(self, base_prompt, combinations, predictor, evaluator, dataset):
        """
        評估不同 few-shot 組合的效果
        
        Args:
            base_prompt: 基礎 prompt
            combinations: few-shot 組合列表
            predictor: 預測器
            evaluator: 評估器
            dataset: 測試資料集
            
        Returns:
            dict: 最佳組合的資訊 {'combination': DataFrame, 'score': float, 'prompt': str}
        """
        if not self.enable_few_shot:
            return {'combination': pd.DataFrame(), 'score': 0.0, 'prompt': base_prompt}
            
        best_combination = pd.DataFrame()
        best_score = -1
        best_prompt = base_prompt
        
        # 備份原始的 records 和 predictor 指令
        original_records = dataset.records.copy()
        original_instruct = predictor.cur_instruct
        
        logging.info(f"開始評估 {len(combinations)} 種 few-shot 組合...")
        
        for i, combination in enumerate(combinations):
            try:
                # 恢復原始狀態
                dataset.records = original_records.copy()
                
                # 結合 prompt 與 few-shot
                combined_prompt = self.combine_prompt_with_few_shot(base_prompt, combination)
                
                # 設定預測器的 prompt
                predictor.cur_instruct = combined_prompt
                
                # 進行預測
                records = predictor.apply(dataset, 0, leq=True)
                dataset.update(records)
                
                # 評估
                evaluator.dataset = dataset.get_leq(0)
                evaluator.eval_score()
                score = evaluator.mean_score
                
                logging.info(f"Few-shot 組合 {i+1}/{len(combinations)}: 分數 = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_combination = combination.copy()
                    best_prompt = combined_prompt
                    
            except Exception as e:
                logging.warning(f"評估 few-shot 組合 {i+1} 時出錯: {e}")
                continue
        
        # 恢復原始狀態
        dataset.records = original_records
        predictor.cur_instruct = original_instruct
        
        result = {
            'combination': best_combination,
            'score': best_score,
            'prompt': best_prompt
        }
        
        logging.info(f"最佳 few-shot 組合分數: {best_score:.4f}")
        return result
    
    def save_best_few_shot(self, prompt_id, best_result):
        """
        儲存最佳 few-shot 組合
        
        Args:
            prompt_id: prompt 的 ID 或描述
            best_result: evaluate_few_shot_combinations 的回傳結果
        """
        self.best_few_shots[prompt_id] = best_result
        
        # 儲存到檔案
        output_file = self.output_dir / 'best_few_shots.json'
        
        # 準備要儲存的資料
        save_data = {}
        for pid, result in self.best_few_shots.items():
            # 先將 NA/NaN 轉成 None
            if not result['combination'].empty:
                safe_records = result['combination'].where(pd.notnull(result['combination']), None).to_dict('records')
            else:
                safe_records = []
            save_data[pid] = {
                'score': result['score'],
                'prompt': result['prompt'],
                'few_shot_examples': safe_records
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"最佳 few-shot 組合已儲存至: {output_file}")
    
    def load_best_few_shot(self):
        """載入已儲存的最佳 few-shot 組合"""
        output_file = self.output_dir / 'best_few_shots.json'
        
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
                
            for pid, data in save_data.items():
                self.best_few_shots[pid] = {
                    'score': data['score'],
                    'prompt': data['prompt'],
                    'combination': pd.DataFrame(data['few_shot_examples'])
                }
                
            logging.info(f"已載入 {len(self.best_few_shots)} 個最佳 few-shot 組合")
        
    def get_best_prompt_for_optimization(self, base_prompt):
        """
        取得用於優化的 prompt（不包含 few-shot）
        
        Args:
            base_prompt: 基礎 prompt
            
        Returns:
            str: 用於優化的純 prompt
        """
        return base_prompt  # 優化時不使用 few-shot 