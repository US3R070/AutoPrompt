from optimization_pipeline import OptimizationPipeline
import logging
import json

class GenOptimizationPipeline(OptimizationPipeline):
    def get_few_shot_examples(self, max_examples=5):
        # 只取 label==5 的 few-shot 範例
        if 'label' not in self.dataset.records.columns or 'text' not in self.dataset.records.columns or 'answer' not in self.dataset.records.columns:
            return ''
        few_shot_df = self.dataset.records[self.dataset.records['label'] == 5]
        few_shot_df = few_shot_df.drop_duplicates(subset=['text', 'answer'])
        few_shot_df = few_shot_df.head(max_examples)
        examples = ["範例："]
        for _, row in few_shot_df.iterrows():
            examples.append(f"\n###User input:\n{row['text']}\n###Model prediction:\n{row['answer']}")
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
            "error_analysis": error_analysis
        }
        print("prompt_input : ",prompt_input)
        
        # 產生新的提示詞，並加入few shot，假如需要NO_THINK則加入NO_THINK 
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        if "NO_THINK" in self.cur_prompt:
            self.cur_prompt = prompt_suggestion['prompt']+"\n\n"+"NO_THINK"
        else :
            self.cur_prompt = prompt_suggestion['prompt']
        self.cur_prompt = self.cur_prompt + "\n\n" + few_shot_block
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
        # 只在 batch_id > 0 時才執行 annotator
        if self.batch_id > 0 or generated:
            logging.info(f'Running annotator on new samples for batch_id: {self.batch_id}')
            self.annotator.cur_instruct = self.cur_prompt
            records = self.annotator.apply(self.dataset, self.batch_id)
            self.dataset.update(records)
        else:
            logging.info('Skipping annotator for initial dataset (batch_id=0).')
        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor')
        records = self.predictor.apply(self.dataset, self.batch_id, leq=True)
        
        self.dataset.update(records)
        
        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        # if self.batch_id > 0 or generated:
        #     self.eval.eval_score()
        self.eval.eval_score()
        for idx, row in self.eval.dataset.iterrows():
            self.dataset.records.loc[self.dataset.records['id'] == row['id'], 'score'] = row['score']
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        # print("large_errors : ",large_errors)
        # print("text : ",self.dataset.records['text'], " annotation : ",self.dataset.records['annotation'], " model_predicts : ",self.dataset.records['prediction'], " score : ",self.dataset.records['score'])
        
        print("self.dataset.records : ",self.dataset.records)
        
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