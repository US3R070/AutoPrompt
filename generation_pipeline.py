from optimization_pipeline import OptimizationPipeline
import logging
import json
import pandas as pd
from estimator import give_estimator
import numpy as np

class GenOptimizationPipeline(OptimizationPipeline):
    def __init__(self, config, task_description, initial_prompt, output_path, classifier_config, ranker_config, best_classifier_prompt, best_ranker_prompt,output_format_definition, **kwargs):
        super().__init__(config, task_description, initial_prompt, output_path, **kwargs)
        self.output_format_definition = output_format_definition
        self.prompt_for_history = self.cur_prompt

        # Setup Classifier
        self.classifier_predictor_obj = classifier_config.predictor
        self.classifier_predictor_obj.config.instruction = best_classifier_prompt['prompt']
        if 'few_shot_examples' in best_classifier_prompt:
            self.classifier_predictor_obj.config.few_shot_examples = best_classifier_prompt['few_shot_examples']
        self.classifier_predictor = give_estimator(self.classifier_predictor_obj)
        self.classifier_predictor.init_chain(label_schema=classifier_config.dataset.label_schema)
        
        # Setup Ranker
        self.ranker_predictor_obj = ranker_config.predictor
        self.ranker_predictor_obj.config.instruction = best_ranker_prompt['prompt']
        if 'few_shot_examples' in best_ranker_prompt:
            self.ranker_predictor_obj.config.few_shot_examples = best_ranker_prompt['few_shot_examples']
        self.ranker_predictor = give_estimator(self.ranker_predictor_obj)
        self.ranker_predictor.init_chain(label_schema=ranker_config.dataset.label_schema)

    def _evaluate_generation_step(self):
        records = self.dataset.get_leq(self.batch_id)

        # --- Step 1: Classification ---
        classifier_df_data = []
        for i, record in records.iterrows():
            classifier_df_data.append({
                'id': record['id'],
                'text': f"---Input: {record['text']}---Model Output{record['prediction']}"
            })
        
        classifier_df = pd.DataFrame(classifier_df_data)
        class_results = self.classifier_predictor.apply_dataframe(classifier_df)

        # --- Step 2: Ranking (for those that passed classification) ---
        if 'score' not in self.dataset.records.columns:
            self.dataset.records['score'] = 1.0
        if 'feedback' not in self.dataset.records.columns:
            self.dataset.records['feedback'] = ''
        
        ranker_df_data = []
        ranker_indices_map = {} 
        
        for i, row in class_results.iterrows():
            original_id = row['id']
            if str(row['prediction']).lower() == 'true':
                ranker_indices_map[len(ranker_df_data)] = original_id
                original_record = records[records['id'] == original_id].iloc[0]
                ranker_df_data.append({
                    'id': len(ranker_df_data),
                    'text': f"---Input: {original_record['text']}---Model Output:{original_record['prediction']}"
                })
            else:
                self.dataset.records.loc[self.dataset.records['id'] == original_id, 'score'] = 1.0
                self.dataset.records.loc[self.dataset.records['id'] == original_id, 'feedback'] = "Classifier Failed: The output format was incorrect and did not follow the rules."

        if ranker_df_data:
            ranker_df = pd.DataFrame(ranker_df_data)
            rank_results = self.ranker_predictor.apply_dataframe(ranker_df)
            
            for _, rank_row in rank_results.iterrows():
                original_id = ranker_indices_map[rank_row['id']]
                try:
                    score = float(rank_row['prediction'])
                except (ValueError, TypeError):
                    score = 1.0  # Assign a default low score if conversion fails
                self.dataset.records.loc[self.dataset.records['id'] == original_id, 'score'] = score
                self.dataset.records.loc[self.dataset.records['id'] == original_id, 'feedback'] = "" 
            print("self.dataset.records : ",self.dataset.records)

        self.eval.mean_score = self.dataset.records[self.dataset.records['batch_id'] <= self.batch_id]['score'].mean()

    def get_few_shot_examples(self, max_examples=5):
        # 只取 label==5 的 few-shot 範例
        if 'label' not in self.dataset.records.columns or 'text' not in self.dataset.records.columns or 'answer' not in self.dataset.records.columns:
            return ''
        few_shot_df = self.dataset.records[self.dataset.records['label'] == 5]
        few_shot_df = few_shot_df.drop_duplicates(subset=['text', 'answer'])
        few_shot_df = few_shot_df.head(max_examples)
        examples = [""]
        for _, row in few_shot_df.iterrows():
            examples.append(f"---\nInput: {row['text']}\n---\n{row['answer']}\n")
        examples.append(f"---\nInput: ")
        return "".join(examples)

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
        # 取得 error_analysis
        error_analysis = last_history[-1].get('analysis', '') if last_history else ''
        # 新增 few-shot block
        few_shot_block = self.get_few_shot_examples(max_examples=self.config.few_shot_examples)
        prompt_input = {
            "history": history_prompt.replace('\n', '').replace('#', ''),
            "task_description": self.task_description,
            "prompt": self.cur_prompt,
            "labels": labels,
            "error_analysis": error_analysis,
            "current_prompt_for_constraint_analysis": self.cur_prompt, # 新增：用於約束分析的當前提示詞
            "output_format_definition": self.output_format_definition # 新增：將黃金模板作為參考傳遞
        }
        #print("prompt_input : ",prompt_input)
        
        # 產生新的提示詞，並加入few shot，假如需要NO_THINK則加入NO_THINK
        prompt_suggestion = self.meta_chain.step_prompt_chain.invoke(prompt_input)
        
        print("prompt_suggestion : ",prompt_suggestion)
        
        if 'text' in prompt_suggestion:
            new_prompt = prompt_suggestion['text']
        elif 'prompt' in prompt_suggestion:
            new_prompt = prompt_suggestion['prompt']
        else:
            raise ValueError("prompt_suggestion 中沒有 'prompt' 或 'text' 鍵")
        
        if "NO_THINK" in self.cur_prompt:
            self.cur_prompt = new_prompt + "\n\n" + "NO_THINK"
        else:
            self.cur_prompt = new_prompt
        
        self.cur_prompt = self.cur_prompt + "\n\n" + few_shot_block
        self.log_and_print(f'Get new prompt:\n{self.cur_prompt}')
        
        self.batch_id += 1
        if len(self.dataset) < self.config.dataset.max_samples:
            batch_input = {
                "num_samples": self.config.meta_prompts.samples_generation_batch,
                "task_description": self.task_description+"\n\n"+"務必用繁體中文生成問題",
                "prompt": self.cur_prompt
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

        if (self.batch_id > 0) or generated:
            logging.info(f'Running annotator on new samples for batch_id: {self.batch_id}')
            self.annotator.cur_instruct = self.cur_prompt
            records = self.annotator.apply(self.dataset, self.batch_id)
            self.dataset.update(records)
        else:
            logging.info('Skipping annotator for initial dataset (batch_id=0).')

        self.predictor.cur_instruct = self.cur_prompt
        logging.info('Running Predictor (raw prompt + single generation with template)')
        
        current_records_df = self.dataset.get_leq(self.batch_id).copy()
        
        if self.predictor.chain is None:
            self.predictor.init_chain(self.dataset.label_schema)

        llm = self.predictor.chain.llm
        
        from langchain_core.prompts import PromptTemplate
        template = PromptTemplate.from_template("{prompt}---Input: {user_input}---")
        
        from tqdm import tqdm
        updated_predictions = []
        print("\n--- Starting Generation (逐筆) ---")
        for index, row in tqdm(current_records_df.iterrows(), total=len(current_records_df), desc="Generating Predictions"):
            try:
                full_input = template.invoke({"prompt": self.cur_prompt, "user_input": row['text']})
                
                response = llm.invoke(full_input)
                
                if hasattr(response, 'content'):
                    prediction = response.content
                else:
                    prediction = str(response)
                
                updated_predictions.append({'id': row['id'], 'prediction': prediction, 'user_input': row['text']})
            except Exception as e:
                logging.error(f"Error during generation for sample {row['id']}: {e}")
                updated_predictions.append({'id': row['id'], 'prediction': '', 'user_input': row['text']})
        print("--- Generation Finished ---\n")

        for pred_data in updated_predictions:
            self.dataset.records.loc[self.dataset.records['id'] == pred_data['id'], 'prediction'] = pred_data['prediction']

        print("\n--- Generated Q&A Pairs ---")
        current_records = self.dataset.get_leq(self.batch_id)
        for index, row in current_records.iterrows():
            if row['prediction'] and pd.notna(row['prediction']):
                print(f"  - Q: {row['text']}")
                print(f"    A: \033[96m{row['prediction']}\033[0m")
                print("  ---")
        print("--- End of Q&A Pairs ---\n")
        
        self._evaluate_generation_step()
             
        logging.info('Calculating Score')
        self.eval.dataset = self.dataset.get_leq(self.batch_id)
        logging.info('Extracting errors based on ranker score <= 3')
        large_errors = self.eval.dataset[self.eval.dataset['score'] <= 3]
        self.eval.errors = large_errors  # Set the errors attribute on the eval object

        # Stop if no errors are found after the first iteration
        if len(large_errors) == 0 and self.batch_id > 0:
            self.log_and_print('No errors found with score <= 3. The current prompt is considered optimal. Stopping optimization.')
            self.eval.add_history(self.prompt_for_history, self.task_description)
            self.save_state()
            return True
        
        print("self.dataset.records : ",self.dataset.records)
        
        # --- Start of Custom Injection Logic ---
        # Temporarily modify the analyzer's prompt template to include the golden template
        original_template = self.eval.analyzer.chain.prompt.template
        try:
            # Use partial to pre-fill the output_format_definition
            # This creates a new prompt template with the value already embedded
            partial_prompt = self.eval.analyzer.chain.prompt.partial(
                output_format_definition=self.output_format_definition
            )
            # Temporarily assign the new, pre-filled prompt to the chain
            self.eval.analyzer.chain.prompt = partial_prompt
            
            # Now, call add_history. It will use the modified prompt.
            self.eval.add_history(self.prompt_for_history, self.task_description)

        finally:
            # IMPORTANT: Always restore the original template to avoid side effects
            self.eval.analyzer.chain.prompt.template = original_template
        # --- End of Custom Injection Logic ---

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