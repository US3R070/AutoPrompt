from optimization_pipeline import OptimizationPipeline
import logging
import json
import pandas as pd

class GenOptimizationPipeline(OptimizationPipeline):
    def __init__(self, *args, classifier_eval_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_eval_config = classifier_eval_config
        self.prompt_for_history = self.cur_prompt

    def get_few_shot_examples(self, max_examples=5):
        # 只取 label==5 的 few-shot 範例
        if 'label' not in self.dataset.records.columns or 'text' not in self.dataset.records.columns or 'answer' not in self.dataset.records.columns:
            return ''
        few_shot_df = self.dataset.records[self.dataset.records['label'] == 5]
        few_shot_df = few_shot_df.drop_duplicates(subset=['text', 'answer'])
        few_shot_df = few_shot_df.head(max_examples)
        examples = ["範例："]
        for _, row in few_shot_df.iterrows():
            examples.append(f"---\nInput:\n {row['text']}\n---\n{row['answer']}")
        examples.append(f"---\nInput:\n ")
        return "\n".join(examples)

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
        # 新增 few-shot block
        few_shot_block = self.get_few_shot_examples(max_examples=self.config.few_shot_examples)
        prompt_input = {
            "history": history_prompt.replace('\n', '').replace('#', ''),
            "task_description": self.task_description,
            "prompt": self.cur_prompt,
            "labels": labels,
            "error_analysis": error_analysis,
            "current_prompt_for_constraint_analysis": self.cur_prompt # 新增：用於約束分析的當前提示詞
        }
        print("prompt_input : ",prompt_input)
        
        # 產生新的提示詞，並加入few shot，假如需要NO_THINK則加入NO_THINK
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        if "NO_THINK" in self.cur_prompt:
            self.prompt_for_history = prompt_suggestion['prompt'] + "\n\n" + "NO_THINK"
        else:
            self.prompt_for_history = prompt_suggestion['prompt']
        
        self.cur_prompt = self.prompt_for_history + "\n\n" + few_shot_block
        self.log_and_print(f'Get new prompt:\n{self.cur_prompt}')
        
        self.batch_id += 1
        if len(self.dataset) < self.config.dataset.max_samples:
            batch_input = {
                "num_samples": self.config.meta_prompts.samples_generation_batch,
                "task_description": self.task_description+"\n\n"+"務必用繁體中文生成問題",
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
            # 改為逐筆 invoke
            # samples_batches = self.meta_chain.step_samples.batch_invoke(batch_inputs, self.config.meta_prompts.num_workers)
            # new_samples = [element for sublist in samples_batches for element in sublist['samples']]
            from tqdm import tqdm
            new_samples = []
            print("\n--- Starting Sample Generation (逐筆) ---")
            for single_input in tqdm(batch_inputs, desc="Generating Samples"):
                try:
                    result = self.meta_chain.step_samples.invoke(single_input)
                    if isinstance(result, dict) and 'samples' in result:
                        new_samples.extend(result['samples'])
                    else:
                        logging.warning(f"Unexpected result format from sample generation: {result}")
                except Exception as e:
                    logging.error(f"Error during single sample generation: {e}")
            print("--- Sample Generation Finished ---\n")
            # new_samples = [f"請用中文生成：{sample}" for sample in new_samples]
            new_samples = self.dataset.remove_duplicates(new_samples)
            self.dataset.add(new_samples, self.batch_id)

    def evaluate_with_classifier_and_ranker(self):
        """使用classifier和ranker進行評估"""
        from utils.llm_chain import get_llm
        from estimator.estimator_llm import LLMEstimator
        import re
        
        testdata = self.dataset.get_leq(self.batch_id)
        
        # Classifier 評估 (改為逐筆 invoke 並顯示進度)
        classifier_scores = []
        if self.classifier_eval_config:
            from tqdm import tqdm
            classifier_estimator = LLMEstimator(self.classifier_eval_config)
            classifier_estimator.init_chain(self.classifier_eval_config.label_schema)
            
            # 根據評估設定，直接獲取 LLM 實例，這才是正確的做法
            classifier_llm = get_llm(self.classifier_eval_config.llm)
            
            classifier_df = testdata.copy()
            
            # 獲取包含指令和 few-shot 的完整 prompt 內容
            # 這是分類器能正確運作的關鍵
            full_prompt_template = self.classifier_eval_config.instruction
            
            print("\n--- Starting Classifier Evaluation (逐筆) ---")
            progress_bar = tqdm(total=len(classifier_df), desc="Classifying", unit="sample")
            for index, row in classifier_df.iterrows():
                try:
                    # 準備單一樣本的輸入
                    sample_input = f"---\nInput:\n {row['text']}\n---\n{row['answer']}"
                    
                    # 將 instruction、few-shot 和當前樣本的輸入手動組合成完整的 prompt
                    final_prompt = f"{full_prompt_template}\n\n{sample_input}"

                    # 直接使用正確獲取的 llm 實例來 invoke，而不是透過 estimator
                    response = classifier_llm.invoke(final_prompt)
                    
                    # 處理回應
                    if isinstance(response, dict) and 'text' in response:
                        classifier_result_text = response['text'].strip().lower()
                    else:
                        classifier_result_text = str(response).strip().lower()

                    # 從回應中解析出標籤 (True/False)
                    label_schema = self.classifier_eval_config.label_schema
                    pattern = r'(' + '|'.join(re.escape(label) for label in label_schema) + r')'
                    match = re.search(pattern, classifier_result_text, re.IGNORECASE)
                    
                    is_compliant = False
                    if match:
                        # 判斷是否為 'True'
                        is_compliant = match.group(1).lower() == 'true'

                    classifier_scores.append(1 if is_compliant else 0)
                    
                    # # Print 結果
                    # status = "\033[92mPass\033[0m" if is_compliant else "\033[91mFail\033[0m"
                    # parsed_label = match.group(1) if match else "N/A"
                    # print(f"  - Sample {index}: [ {status} ] - Prediction: '{row['prediction'][:80]}' - Classifier says: '{parsed_label}'")

                except Exception as e:
                    
                    #logging.error(f"Error during classifier invocation for sample {index}: {e}")
                    # classifier_scores.append(0) # 發生錯誤視為不通過
                    print(f"  - Sample {index}: [ \033[91mError\033[0m ] - Prediction: '{row['prediction'][:80]}' - Exception: {e}\n")

                progress_bar.update(1)
            
            progress_bar.close()
            print("--- Classifier Evaluation Finished ---\n")
        else:
            classifier_scores = [1] * len(testdata)  # 如果沒有classifier配置，預設為通過
        
        # Ranker 評估
        ranker_scores = [1] * len(testdata)  # 初始化所有樣本的 ranker 分數為 1
        
        # 篩選出 classifier 評估為 1 的樣本，這些才需要進行 ranker 評估
        passed_classifier_indices = [i for i, score in enumerate(classifier_scores) if score == 1]
        
        if passed_classifier_indices:
            ranker_estimator = LLMEstimator(self.eval.function_params)
            ranker_estimator.init_chain(self.eval.function_params.label_schema)
            
            # 創建 ranker 評估用的 dataframe，只包含通過 classifier 的樣本
            ranker_df = testdata.iloc[passed_classifier_indices].copy()
            ranker_df['text'] = ranker_df.apply(lambda row: f"---\nInput:\n {row['text']}\n---\n{row['answer']}", axis=1)
            
            print("\n--- Starting Ranker Evaluation (Batch) ---")
            ranker_results = ranker_estimator.apply_dataframe(ranker_df)
            print("--- Ranker Evaluation Finished ---\n")

            # 將 ranker 結果映射回原始的 ranker_scores 列表
            for i, row in ranker_results.iterrows():
                original_index = row['id'] # 假設 ranker_df 的 id 欄位保留了原始 testdata 的索引
                ranker_result = row['prediction']
                try:
                    score = int(ranker_result.strip())
                    score = max(1, min(5, score))  # 確保分數在1-5範圍內
                except ValueError:
                    score = 1  # 預設分數
                ranker_scores[original_index] = score
        
        # 計算綜合分數：如果classifier不通過，直接給1分；否則使用ranker分數
        combined_scores = []
        for i in range(len(testdata)):
            if classifier_scores[i] == 0:  # 不符合硬性規定
                combined_scores.append(1)
            else:
                combined_scores.append(ranker_scores[i])
        
        # 更新dataset的score
        for i, row in testdata.iterrows():
            self.dataset.records.loc[self.dataset.records['id'] == row['id'], 'score'] = combined_scores[i-1] if i > 0 else combined_scores[0]
        
        # 計算平均分數
        mean_score = sum(combined_scores) / len(combined_scores)
        self.eval.mean_score = mean_score
        
        print(f"Classifier 通過率: {sum(classifier_scores)/len(classifier_scores):.2f}")
        print(f"Ranker 平均分數: {sum(ranker_scores)/len(ranker_scores):.2f}")
        print(f"綜合平均分數: {mean_score:.2f}")
        
        return mean_score

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

        # 只在以下情況執行 annotator：
        # 1. batch_id > 0 且沒有 ground truth 資料，或
        # 2. 有新生成的資料 (generated=True)
        if (self.batch_id > 0) or generated:
            logging.info(f'Running annotator on new samples for batch_id: {self.batch_id}')
            self.annotator.cur_instruct = self.cur_prompt
            records = self.annotator.apply(self.dataset, self.batch_id)
            self.dataset.update(records)
        else:
            logging.info('Skipping annotator for initial dataset (batch_id=0).')

        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor (raw prompt + single generation with template)')
        
        # 獲取當前批次的數據
        current_records_df = self.dataset.get_leq(self.batch_id).copy()
        
        # 初始化 predictor 的 chain (如果尚未初始化)
        if self.predictor.chain is None:
            self.predictor.init_chain(self.dataset.label_schema)

        # 獲取 LLM 實例
        llm = self.predictor.chain.llm
        
        # 定義 PromptTemplate
        from langchain_core.prompts import PromptTemplate
        template = PromptTemplate.from_template("{prompt}---\nInput: {user_input}\n---")
        
        # 逐筆處理數據，使用 raw prompt 進行生成
        from tqdm import tqdm
        updated_predictions = []
        print("\n--- Starting Generation (逐筆) ---")
        for index, row in tqdm(current_records_df.iterrows(), total=len(current_records_df), desc="Generating Predictions"):
            try:
                # 組合 prompt 和 user_input
                full_input = template.invoke({"prompt": self.cur_prompt, "user_input": row['text']})
                
                # 調用 LLM 進行生成
                response = llm.invoke(full_input)
                
                # 從回應中提取 content 欄位
                if hasattr(response, 'content'):
                    prediction = response.content
                else:
                    prediction = str(response) # Fallback if content attribute is not found
                
                updated_predictions.append({'id': row['id'], 'prediction': prediction})
            except Exception as e:
                logging.error(f"Error during generation for sample {row['id']}: {e}")
                updated_predictions.append({'id': row['id'], 'prediction': ''}) # 發生錯誤時給空字串或預設值
        print("--- Generation Finished ---\n")

        # 將更新後的預測結果合併回原始數據集
        for pred_data in updated_predictions:
            self.dataset.records.loc[self.dataset.records['id'] == pred_data['id'], 'prediction'] = pred_data['prediction']

        # 由於我們直接更新了 self.dataset.records，這裡不需要再次調用 self.dataset.update(records)
        # 但為了保持一致性，可以確保 self.dataset.records 已經包含了所有更新
        # self.dataset.update(self.dataset.records) # 這行可能不需要，取決於 dataset 內部實現

        # --- 新增：列印生成的 題目和答案 配對 ---
        print("\n--- Generated Q&A Pairs ---")
        # 從更新後的 records DataFrame 中提取 text (題目) 和 prediction (答案)
        # 我們只關心當前這輪 (leq=True) 生成或更新的樣本
        current_records = self.dataset.get_leq(self.batch_id)
        for index, row in current_records.iterrows():
            # 確保 prediction 不是 None 或 NaN
            if row['prediction'] and pd.notna(row['prediction']):
                print(f"  - Q: {row['text']}")
                print(f"    A: \033[96m{row['prediction']}\033[0m")
                print("  ---")
        print("--- End of Q&A Pairs ---\n")
        # --- 結束新增 ---
        
        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        
        # 使用新的評估機制
        if self.classifier_eval_config:
            self.evaluate_with_classifier_and_ranker()
        else:
            # 如果沒有classifier配置，使用原來的評估方式
            self.eval.eval_score()
            for idx, row in self.eval.dataset.iterrows():
                self.dataset.records.loc[self.dataset.records['id'] == row['id'], 'score'] = row['score']
        
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        
        print("self.dataset.records : ",self.dataset.records)
        
        self.eval.add_history(self.prompt_for_history, self.task_description)
        if self.config.use_wandb:
            large_errors = large_errors.sample(n=min(6, len(large_errors)))
            correct_samples = self.eval.extract_correct()
            correct_samples = correct_samples.sample(n=min(6, len(correct_samples)))
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