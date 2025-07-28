import logging
import json
from pathlib import Path
import pandas as pd
import wandb
import math
import re
from sklearn.metrics import confusion_matrix
from optimization_pipeline import OptimizationPipeline
from dataset.base_dataset import DatasetBase
from utils.llm_chain import get_llm


class SingleClassifyOptimizationPipeline(OptimizationPipeline):
    def __init__(self, *args, meta_chain=None, few_shot_selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_chain = meta_chain
        self.few_shot_selector = few_shot_selector
        # 確保predictor與其他步驟兼容
        self._ensure_predictor_compatibility()

    def _ensure_predictor_compatibility(self):
        """
        確保predictor與其他步驟兼容
        """
        if hasattr(self, 'predictor') and self.predictor is not None:
            # 確保predictor有cur_instruct屬性
            if not hasattr(self.predictor, 'cur_instruct'):
                self.predictor.cur_instruct = None
            
            # 確保predictor有init_chain方法（如果需要的話）
            if not hasattr(self.predictor, 'init_chain'):
                def init_chain(label_schema):
                    # 空實現，因為我們使用raw prompt
                    pass
                self.predictor.init_chain = init_chain

    def step(self, current_iter, total_iter):
        
        generated = False
        
        # 確保predictor有正確的初始化
        if not hasattr(self.predictor, 'cur_instruct') or self.predictor.cur_instruct is None:
            self.predictor.cur_instruct = self.cur_prompt
        
        self.log_and_print(f'Starting step {self.batch_id}')
        if len(self.dataset.records) == 0:
            self.log_and_print('Dataset is empty generating initial samples')
            self.generate_initial_samples()
            generated = True
        print("self.dataset.records : ",self.dataset.records)
        if self.config.use_wandb:
            cur_batch = self.dataset.get_leq(self.batch_id)
            random_subset = cur_batch.sample(n=min(10, len(cur_batch)))[['text']]
            self.wandb_run.log(
                {"Prompt": wandb.Html(f"<p>{self.cur_prompt}</p>"), "Samples": wandb.Table(dataframe=random_subset)},
                step=self.batch_id)

        # 檢查是否有 ground truth 資料（非合成資料）
        has_ground_truth = len(self.dataset.records) > 0 and 'is_synthetic' in self.dataset.records.columns
        ground_truth_count = 0
        if has_ground_truth:
            ground_truth_count = len(self.dataset.records[self.dataset.records['is_synthetic'] == False])
        
        # 只在以下情況執行 annotator：
        # 1. batch_id > 0 且沒有 ground truth 資料，或
        # 2. 有新生成的資料 (generated=True)
        if (self.batch_id > 0 and ground_truth_count == 0) or generated:
            logging.info(f'Running annotator on new samples for batch_id: {self.batch_id}')
            self.annotator.cur_instruct = self.cur_prompt
            records = self.annotator.apply(self.dataset, self.batch_id)
            self.dataset.update(records)
        else:
            if ground_truth_count > 0:
                logging.info(f'Skipping annotator for batch_id: {self.batch_id} - using {ground_truth_count} ground truth samples')
            else:
                logging.info('Skipping annotator for initial dataset (batch_id=0).')

        # Few-shot 評估（如果啟用）
        if self.few_shot_selector:
            logging.info('開始 Few-shot 評估...')
            max_num = self.dataset.get_leq(0)[self.dataset.get_leq(0)['is_synthetic'] == False].shape[0]
            # 計算所有可能的組合數
            print(f"所有可能的組合數: {math.comb(max_num,self.few_shot_selector.num_shots)}")
            max_num = min(10,math.comb(max_num,self.few_shot_selector.num_shots))
            combinations = self.few_shot_selector.sample_few_shot_combinations(max_combinations=max_num)
            best_few_shot_result = self.few_shot_selector.evaluate_few_shot_combinations(
                self.cur_prompt, combinations, self.predictor, self.eval, self.dataset
            )
            # predictor 用 best few-shot 組合
            predictor_prompt = best_few_shot_result['prompt']
            prompt_id = f"step_{self.batch_id}_prompt"
            self.few_shot_selector.save_best_few_shot(prompt_id, best_few_shot_result)
            logging.info(f'Few-shot 最佳分數: {best_few_shot_result["score"]:.4f}')
            # 新增：few-shot raw prompt
            raw_prompt = predictor_prompt
        else:
            predictor_prompt = self.cur_prompt
            raw_prompt = self.cur_prompt

        # 使用 raw prompt 進行預測
        logging.info('Running Raw Prompt Evaluation')
        testdata = self.dataset.get_leq(self.batch_id)
        llm = get_llm(self.config.predictor.config.llm)
        raw_predictions = []
        
        # 添加進度條
        from tqdm import tqdm
        progress_bar = tqdm(total=len(testdata), desc="Processing samples", unit="sample")
        
        for i, row in testdata.iterrows():
            try:
                # few-shot prompt 已經在 raw_prompt
                user_input = row['text']
                prompt = f"{raw_prompt}\n\n {user_input}"
                response = llm.invoke(prompt)  # 直接傳 str
                
                # 只抓 True/False
                if isinstance(response, dict) and 'text' in response:
                    resp_text = response['text']
                else:
                    resp_text = str(response)
                
                # 只抓 self.config.dataset.label_schema 的 label
                label_schema = self.config.dataset.label_schema
                pattern = r'(' + '|'.join(re.escape(label) for label in label_schema) + r')'
                match = re.search(pattern, resp_text, re.IGNORECASE)
                label = match.group(1) if match else 'Discarded'
                raw_predictions.append(label)
                
            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                raw_predictions.append('Discarded')
            
            # 更新進度條
            progress_bar.update(1)
        
        # 關閉進度條
        progress_bar.close()
        
        # 更新dataset的prediction欄位
        for i, pred in enumerate(raw_predictions):
            if i < len(self.dataset.records):
                self.dataset.records.iloc[i, self.dataset.records.columns.get_loc('prediction')] = pred
        
        print("self.dataset.records['prediction'] : ",self.dataset.records['prediction'])

        # 設置eval的dataset並計算分數
        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        self.eval.eval_score()
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        
        # 記錄到 history
        score = self.eval.history[-1]['score'] if self.eval.history else 0.0
        self.eval.history.append({
            'prompt': raw_prompt,
            'score': score,
            'errors': large_errors,
            'confusion_matrix': confusion_matrix(self.eval.dataset['annotation'].astype(str).str.lower(), 
                                              self.eval.dataset['prediction'].astype(str).str.lower()),
            'analysis': '[raw prompt 真實評分]'
        })
        # ====== END ======

        if self.config.use_wandb:
            large_errors = large_errors.sample(n=min(6, len(large_errors)))
            correct_samples = self.eval.extract_correct()
            correct_samples = correct_samples.sample(n=min(6, len(correct_samples)))
            vis_data = pd.concat([large_errors, correct_samples])
            wandb_log_data = {"score": self.eval.history[-1]['score'],
                            "prediction_result": wandb.Table(dataframe=vis_data),
                            'Total usage': self.calc_usage()}
            self.wandb_run.log(wandb_log_data, step=self.batch_id)

        if self.stop_criteria():
            self.log_and_print('Stop criteria reached')
            return True
        if current_iter != total_iter-1:
            self.run_step_prompt()
        self.save_state()

        def save_prompt_history(output_dir, batch_id, prompt, score):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / "best_prompts_history.jsonl"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "batch_id": batch_id,
                    "prompt": prompt,
                    "score": score
                }, ensure_ascii=False) + "\n")
        save_prompt_history(self.output_path, self.batch_id, raw_prompt, score)

        return False

    def run_step_prompt(self):
        # 確保predictor的cur_instruct有正確的值
        if not hasattr(self.predictor, 'cur_instruct') or self.predictor.cur_instruct is None:
            self.predictor.cur_instruct = self.cur_prompt
        
        # 只取上一輪（目前最佳 prompt）產生的錯誤樣本
        last_history = [self.eval.history[-1]] if self.eval.history else []
        error_samples = []
        for sample in last_history:
            if 'errors' in sample and isinstance(sample['errors'], pd.DataFrame):
                for _, row in sample['errors'].iterrows():
                    error_samples.append({
                        'input': row.get('text', ''),
                        'prediction': row.get('prediction', ''),
                        'label': row.get('annotation', '')
                    })
        # 用 meta_chain.error_analysis 產生 analysis
        analysis = None
        if self.meta_chain is not None and error_samples:
            # 準備 error_analysis prompt_input
            large_error_to_str = self.eval.large_error_to_str(last_history[-1]['errors'], self.config.meta_prompts.num_err_prompt)
            prompt_input = {
                'task_description': self.task_description,
                'accuracy': self.eval.mean_score,
                'prompt': self.cur_prompt,
                'failure_cases': large_error_to_str
            }
            
            # 若有 label_schema 也帶入
            if 'label_schema' in self.config.dataset.keys():
                prompt_input['labels'] = json.dumps(self.config.dataset.label_schema)
            # 若有混淆矩陣也帶入
            if 'confusion_matrix' in last_history[-1]:
                prompt_input['confusion_matrix'] = last_history[-1]['confusion_matrix']
            
            # 用error_analysis找問題
            analysis_result = self.meta_chain.error_analysis.invoke(prompt_input)
            analysis = analysis_result['text'] if isinstance(analysis_result, dict) and 'text' in analysis_result else str(analysis_result)
            print("[Error Analysis 分析結果]:\n" + analysis)
        
        # 將 analysis 傳給 meta_chain.step_prompt_chain
        history_prompt = '\n'.join([self.eval.sample_to_text(sample,
                                                            num_errors_per_label=self.config.meta_prompts.num_err_prompt,
                                                            is_score=True) for sample in last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_description,
                        'error_analysis': analysis or (last_history[-1].get('analysis', '') if last_history else '')}
        

        if 'label_schema' in self.config.dataset.keys():
            prompt_input["labels"] = json.dumps(self.config.dataset.label_schema)
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        
        # 檢查predictor的cur_instruct是否包含NO_THINK
        if hasattr(self.predictor, 'cur_instruct') and self.predictor.cur_instruct and "NO_THINK" in self.predictor.cur_instruct:
            prompt_suggestion['prompt'] = prompt_suggestion['prompt']+"\n/NO_THINK"
        
        if self.meta_chain.step_prompt_chain.llm_config.type == 'google':
            if isinstance(prompt_suggestion, list) and len(prompt_suggestion) == 1:
                prompt_suggestion = prompt_suggestion[0]['args']
        
        self.log_and_print(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        self.log_and_print(f'Get new prompt:\n{prompt_suggestion["prompt"]}')
        self.batch_id += 1
        
        # 更新predictor的cur_instruct
        if hasattr(self.predictor, 'cur_instruct'):
            self.predictor.cur_instruct = prompt_suggestion['prompt']
        
        if len(self.dataset) < self.config.dataset.max_samples:
            batch_input = {"num_samples": self.config.meta_prompts.samples_generation_batch,
                           "task_description": self.task_description,
                           "prompt": prompt_suggestion['prompt']}
            batch_inputs = self.generate_samples_batch(batch_input, self.config.meta_prompts.num_generated_samples,
                                                       self.config.meta_prompts.samples_generation_batch)

            if sum([len(t['errors']) for t in last_history]) > 0:
                history_samples = '\n'.join([self.eval.sample_to_text(sample,
                                                                 num_errors_per_label=self.config.meta_prompts.num_err_samples,
                                                                 is_score=False) for sample in last_history])
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                    batch['history'] = history_samples
                    batch['extra_samples'] = extra_samples_text
            else:
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = DatasetBase.samples_to_text(extra_samples)
                    batch['history'] = 'No previous errors information'
                    batch['extra_samples'] = extra_samples_text

            samples_batches = self.meta_chain.step_samples.batch_invoke(batch_inputs,
                                                                         self.config.meta_prompts.num_workers)
            new_samples = [element for sublist in samples_batches for element in sublist['samples']]
            new_samples = self.dataset.remove_duplicates(new_samples)
            self.dataset.add(new_samples, self.batch_id)
            logging.info('Get new samples')
        self.cur_prompt = prompt_suggestion['prompt'] 