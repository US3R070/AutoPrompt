from optimization_pipeline import OptimizationPipeline
import logging
import json

class RnkOptimizationPipeline(OptimizationPipeline):
    def run_step_prompt(self):
        # 產生新一輪的 prompt 與合成資料
        last_history = [self.eval.history[-1]] if self.eval.history else []
        history_prompt = '\n'.join([
            self.eval.sample_to_text(sample, is_score=False) for sample in last_history
        ])
        # 取得 labels
        if hasattr(self.config.dataset, 'label_schema'):
            labels = self.config.dataset.label_schema
        elif 'label_schema' in self.config.dataset:
            labels = self.config.dataset['label_schema']
        else:
            labels = []
        # 取得 error_analysis（仿 classification）
        analysis = None
        if self.meta_chain is not None and last_history:
            prompt_input_analysis = {
                'task_description': self.task_description,
                'accuracy': self.eval.mean_score,
                'prompt': self.cur_prompt,
                'failure_cases': ''
            }
            if labels:
                prompt_input_analysis['labels'] = json.dumps(labels)
            if 'confusion_matrix' in last_history[-1]:
                prompt_input_analysis['confusion_matrix'] = last_history[-1]['confusion_matrix']
            
            analysis_result = self.meta_chain.error_analysis.invoke(prompt_input_analysis)
            analysis = analysis_result['text'] if isinstance(analysis_result, dict) and 'text' in analysis_result else str(analysis_result)
        error_analysis = analysis or (last_history[-1].get('analysis', '') if last_history else '')
        prompt_input = {
            "history": history_prompt.replace('\n', '').replace('#', ''),
            "task_description": self.task_description,
            "prompt": self.cur_prompt,
            "labels": labels,
            "error_analysis": error_analysis
        }
        
        print("prompt_input : ",prompt_input)
        
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        self.cur_prompt = prompt_suggestion['prompt'] + '\n'
        self.log_and_print(f'Get new prompt:\n{self.cur_prompt}')
        # print("after batch_id : ",self.batch_id," self.cur_prompt : ",self.cur_prompt)

        self.batch_id += 1
        if len(self.dataset) < self.config.dataset.max_samples:
            batch_input = {
                "num_samples": self.config.meta_prompts.samples_generation_batch,
                "task_description": self.task_description+"\n\n"+"務必用繁體中文生成",
                "prompt": prompt_suggestion['prompt']
            }
            batch_inputs = self.generate_samples_batch(
                batch_input,
                self.config.meta_prompts.num_generated_samples,
                self.config.meta_prompts.samples_generation_batch
            )
            # 補齊 history/extra_samples
            if sum([len(t['errors']) for t in last_history]) > 0:
                history_samples = '\n'.join([self.eval.sample_to_text(sample, is_score=True) for sample in last_history])
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = type(self.dataset).samples_to_text(extra_samples)
                    batch['history'] = history_samples
                    batch['extra_samples'] = extra_samples_text
            else:
                for batch in batch_inputs:
                    extra_samples = self.dataset.sample_records()
                    extra_samples_text = type(self.dataset).samples_to_text(extra_samples)
                    batch['history'] = 'No previous errors information'
                    batch['extra_samples'] = extra_samples_text
            samples_batches = self.meta_chain.step_samples.batch_invoke(batch_inputs, self.config.meta_prompts.num_workers)
            new_samples = [element for sublist in samples_batches for element in sublist['samples']]
            # new_samples = [f"請用中文生成：{sample}" for sample in new_samples]
            new_samples = self.dataset.remove_duplicates(new_samples)
            self.dataset.add(new_samples, self.batch_id)

    def step(self, current_iter, total_iter):
        self.log_and_print(f'Starting step {self.batch_id}')
        
        generated = False
        
        if len(self.dataset.records) == 0:
            self.log_and_print('Dataset is empty generating initial samples')
            self.generate_initial_samples()
            generated = True
        if self.config.use_wandb:
            cur_batch = self.dataset.get_leq(self.batch_id)
            random_subset = cur_batch.sample(n=min(10, len(cur_batch)))[['text']]
            self.wandb_run.log(
                {"Prompt": f"<p>{self.cur_prompt}</p>", "Samples": random_subset},
                step=self.batch_id)

        # 檢查是否有 ground truth 資料（非合成資料）
        has_ground_truth = 'is_synthetic' in self.dataset.records.columns
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

        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor')
        
        # # 使用 raw prompt 進行預測 (註解掉原始的單筆 invoke 方法)
        # logging.info('Running Raw Prompt Evaluation')
        # testdata = self.dataset.get_leq(self.batch_id)
        # from utils.llm_chain import get_llm
        # import re
        
        # llm = get_llm(self.config.predictor.config.llm)
        # raw_predictions = []
        
        # # 添加進度條
        # from tqdm import tqdm
        # progress_bar = tqdm(total=len(testdata), desc="Processing samples", unit="sample")
        
        # for i, row in testdata.iterrows():
        #     try:
        #         user_input = row['text']
        #         prompt = f"{self.cur_prompt}\n\n {user_input}"
        #         response = llm.invoke(prompt)  # 直接傳 str
                
        #         # 處理回應
        #         if isinstance(response, dict) and 'text' in response:
        #             resp_text = response['text']
        #         else:
        #             resp_text = str(response)
                
        #         # 提取 1-5 分數
        #         label_schema = self.config.dataset.label_schema
        #         pattern = r'\b([1-5])\b'
        #         match = re.search(pattern, resp_text)
        #         label = match.group(1) if match else '1'
        #         raw_predictions.append(label)
                
        #     except Exception as e:
        #         logging.error(f"Error processing sample {i}: {e}")
        #         raw_predictions.append('1')
            
        #     # 更新進度條
        #     progress_bar.update(1)
        
        # # 關閉進度條
        # progress_bar.close()
        
        # # 更新dataset的prediction欄位
        # for i, pred in enumerate(raw_predictions):
        #     if i < len(self.dataset.records):
        #         self.dataset.records.iloc[i, self.dataset.records.columns.get_loc('prediction')] = pred

        # 使用 predictor 進行批次預測
        testdata = self.dataset.get_leq(self.batch_id)
        records = self.predictor.apply(self.dataset, self.batch_id)
        self.dataset.update(records)
        
        # print("self.dataset.records['prediction'] : ",self.dataset.records['prediction'])
        
        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        self.eval.eval_score()
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        print("large_errors : ",large_errors)
        
        self.eval.add_history(self.cur_prompt, self.task_description)
        if self.config.use_wandb:
            large_errors = large_errors.sample(n=min(6, len(large_errors)))
            correct_samples = self.eval.extract_correct()
            correct_samples = correct_samples.sample(n=min(6, len(correct_samples)))
            import pandas as pd
            vis_data = pd.concat([large_errors, correct_samples])
            self.wandb_run.log({"score": self.eval.history[-1]['score'],
                                "prediction_result": vis_data,
                                'Total usage': self.calc_usage()}, step=self.batch_id)
        if self.stop_criteria():
            self.log_and_print('Stop criteria reached')
            return True
        if current_iter != total_iter-1:
            self.run_step_prompt()
        self.save_state()
        return False 