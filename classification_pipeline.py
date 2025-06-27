import logging
import json
from pathlib import Path
import pandas as pd
import wandb
from optimization_pipeline import OptimizationPipeline
from dataset.base_dataset import DatasetBase

class ResOptimizationPipeline(OptimizationPipeline):
    def __init__(self, *args, reasoner=None, few_shot_selector=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoner = reasoner
        self.few_shot_selector = few_shot_selector

    def step(self, current_iter, total_iter):
        self.log_and_print(f'Starting step {self.batch_id}')
        if len(self.dataset.records) == 0:
            self.log_and_print('Dataset is empty generating initial samples')
            self.generate_initial_samples()
        if self.config.use_wandb:
            cur_batch = self.dataset.get_leq(self.batch_id)
            random_subset = cur_batch.sample(n=min(10, len(cur_batch)))[['text']]
            self.wandb_run.log(
                {"Prompt": wandb.Html(f"<p>{self.cur_prompt}</p>"), "Samples": wandb.Table(dataframe=random_subset)},
                step=self.batch_id)

        # 只在 batch_id > 0 時才執行 annotator
        if self.batch_id > 0:
            logging.info(f'Running annotator on new samples for batch_id: {self.batch_id}')
            self.annotator.cur_instruct = self.cur_prompt
            records = self.annotator.apply(self.dataset, self.batch_id)
            self.dataset.update(records)
        else:
            logging.info('Skipping annotator for initial dataset (batch_id=0).')

        # Few-shot 評估（如果啟用）
        if self.few_shot_selector:
            logging.info('開始 Few-shot 評估...')
            combinations = self.few_shot_selector.sample_few_shot_combinations(max_combinations=5)
            # 注意：這裡傳入 base_prompt（不加 few-shot）
            best_few_shot_result = self.few_shot_selector.evaluate_few_shot_combinations(
                self.cur_prompt, combinations, self.predictor, self.eval, self.dataset
            )
            # predictor 用 best few-shot 組合
            predictor_prompt = best_few_shot_result['prompt']
            prompt_id = f"step_{self.batch_id}_prompt"
            self.few_shot_selector.save_best_few_shot(prompt_id, best_few_shot_result)
            logging.info(f'Few-shot 最佳分數: {best_few_shot_result["score"]:.4f}')
        else:
            predictor_prompt = self.cur_prompt

        # predictor 評估（用目前 prompt + best few-shot，如有）
        self.predictor.cur_instruct = predictor_prompt
        logging.info('Running Predictor')
        records = self.predictor.apply(self.dataset, self.batch_id, leq=True)
        self.dataset.update(records)

        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        self.eval.eval_score()
        logging.info('Calculating Score')
        large_errors = self.eval.extract_errors()
        
        
        self.eval.add_history(self.cur_prompt, self.task_description)

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

        # 新增：每一輪都存下當前 prompt 和分數
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
        save_prompt_history(self.output_path, self.batch_id, self.predictor.cur_instruct, self.eval.mean_score)

        return False

    def run_step_prompt(self):
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
        # 呼叫 reasoner
        analysis = None
        if self.reasoner is not None and error_samples:
            print('error_samples:', error_samples)
            analysis = self.reasoner.analyze(self.cur_prompt, error_samples)
            print("[Reasoner 分析結果]:\n" + analysis)
        # 將 analysis 傳給 meta_chain
        history_prompt = '\n'.join([self.eval.sample_to_text(sample,
                                                        num_errors_per_label=self.config.meta_prompts.num_err_prompt,
                                                        is_score=True) for sample in last_history])
        prompt_input = {"history": history_prompt, "task_description": self.task_description,
                        'error_analysis': analysis or (last_history[-1].get('analysis', '') if last_history else '')}
        if 'label_schema' in self.config.dataset.keys():
            prompt_input["labels"] = json.dumps(self.config.dataset.label_schema)
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        prompt_suggestion['prompt'] = prompt_suggestion['prompt']+"\n/NO_THINK"
        if self.meta_chain.step_prompt_chain.llm_config.type == 'google':
            if isinstance(prompt_suggestion, list) and len(prompt_suggestion) == 1:
                prompt_suggestion = prompt_suggestion[0]['args']
        self.log_and_print(f'Previous prompt score:\n{self.eval.mean_score}\n#########\n')
        self.log_and_print(f'Get new prompt:\n{prompt_suggestion["prompt"]}')
        self.batch_id += 1
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