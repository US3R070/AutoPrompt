from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, modify_input_for_ranker, validate_generation_config, override_config
import argparse
import os
from estimator.estimator_llm import LLMEstimator
import pandas as pd
import csv
from pathlib import Path
import numpy as np
from generation_pipeline import GenOptimizationPipeline

def process_dataset(config_params, output_dir, filename,type = 'ranker',):
    dataset_path = Path(config_params.dataset.records_path)
    if dataset_path.is_file():
        df = pd.read_csv(dataset_path)
        modified = False
        # 新增：合併 text 和 answer 欄位
        # if ('text' in df.columns) and ('answer' in df.columns):
        #     df['text'] = df.apply(lambda row: f"Question:{row['text']} , Answer:{row['answer']}", axis=1)
        #     modified = True
        # annotation 欄位直接複製 label
        if 'label' in df.columns:
            df['annotation'] = df['label']
            modified = True
        required_cols = {
            'id': lambda df: range(len(df)),
            'text': '',
            'prediction': pd.NA,
            'metadata': None,
            'annotation': pd.NA,
            'score': pd.NA,
            'batch_id': 0
        }
        if type == 'ranker':
            df['text'] = df.apply(lambda row: f"Question:{row['text']} , Answer:{row['answer']}", axis=1)
            df['text'] = (df['text'] if 'text' in df.columns else '')
            df['annotation'] = (df['label'] if 'label' in df.columns else pd.NA)
        elif type == 'generator':
            df['text'] = (df['text'] if 'text' in df.columns else '')
            # 在生成型任務中，annotation設為期望的最低品質分數
            df['annotation'] = '4'  # 期望分數至少為4
        for col, default in required_cols.items():
            if col not in df.columns:
                if callable(default):
                    df.insert(0 if col == 'id' else len(df.columns), col, default(df))
                else:
                    df[col] = default
                modified = True
        # annotation 欄位全空時自動用 label 填入
        if ('label' in df.columns) and ('annotation' in df.columns) and df['annotation'].isna().all():
            df['annotation'] = df['label'].astype(str)
        if modified:
            processed_dir = Path(output_dir)
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / filename
            df['annotation'] = df['annotation'].astype(str)
            df.to_csv(processed_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            config_params.dataset.records_path = str(processed_path)
            if hasattr(config_params.dataset, 'initial_dataset'):
                config_params.dataset.initial_dataset = str(processed_path)
            print(f"自動添加缺失欄位並寫入: {processed_path}")

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--generation_config_path', default='config/config_diff/config_generation.yml', type=str, help='Configuration file path')
parser.add_argument('--ranker_config_path', default='config/config_diff/config_ranking.yml', type=str, help='Configuration file path')

parser.add_argument('--ranker_task_description',
                    default='你是一個評分者，你必須依據輸入的問題和答案，依照回答的簡潔和具體程度，最多10字，給出1-5分的分數',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--ranker_prompt',
                    default='你是一個評分者，你必須依據輸入的問題和答案，依照回答的簡潔和具體程度，最多10字，給出1-5分的分數',
                    required=False, type=str, help='Prompt to use as initial.')

parser.add_argument('--task_description',
                    default='你是一個回答者，你必須用繁體中文對應輸入，產出一個具體的、最多10字的回答',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--prompt',
                    default='你是一個回答者，你必須對輸入來產出一個最多10字的回答|--輸入 : 你口袋裡有多少錢--輸出 : 62塊錢--輸入 : 你怎麼不去問問神奇海螺呢?輸出 : 我去問問神奇海螺',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_dump', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_ranker_steps', default=1, type=int, help='Number of iterations')
parser.add_argument('--num_generation_steps', default=15, type=int, help='Number of iterations')
parser.add_argument('--has_initial_data', action='store_true', help='資料集是否有初始標註資料（有則 batch_id==0 不做 annotation）')

opt = parser.parse_args()

ranker_config_params = override_config(opt.ranker_config_path)
generation_config_params = override_config(opt.generation_config_path)
validate_generation_config(ranker_config_params, generation_config_params)

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt
    
# 處理 ranking 資料集
process_dataset(ranker_config_params, os.path.join(opt.output_dump, 'ranker'), 'ranking_dataset_processed.csv',type = 'ranker')

ranker_pipeline = GenOptimizationPipeline(ranker_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
if opt.load_dump != '':
    ranker_pipeline.load_state(os.path.join(opt.load_dump, 'ranker'))
    ranker_pipeline.predictor.init_chain(ranker_config_params.dataset.label_schema)

if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
    ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(ranker_config_params, opt.ranker_task_description,
                                                                      opt.ranker_prompt)
    ranker_pipeline.cur_prompt = ranker_mod_prompt
    ranker_pipeline.task_description = ranker_mod_task_desc

best_prompt = ranker_pipeline.run_pipeline(opt.num_ranker_steps)
print("best_prompt for ranker : ",best_prompt)

# 處理 generation 資料集
process_dataset(generation_config_params, os.path.join(opt.output_dump, 'generator'), 'generation_dataset_processed.csv',type = 'generator')

generation_config_params.eval.function_params = ranker_config_params.predictor.config

# print("generation_config_params.eval.function_params : ",generation_config_params.eval.function_params)

generation_config_params.eval.function_params.instruction = best_prompt['prompt']
generation_config_params.eval.function_params.label_schema = ranker_config_params.dataset.label_schema
generation_pipeline = GenOptimizationPipeline(generation_config_params, task_description, initial_prompt,
                                           output_path=os.path.join(opt.output_dump, 'generator'))


if opt.load_dump != '':
    generation_pipeline.load_state(os.path.join(opt.load_dump, 'generator'))

best_generation_prompt = generation_pipeline.run_pipeline(opt.num_generation_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_generation_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_generation_prompt['prompt'] + '\033[0m')

# # 處理 ranking 資料集
# process_dataset(ranker_config_params, os.path.join(opt.output_dump, 'ranker'), 'ranking_dataset_processed.csv')
# # 處理 generation 資料集
# process_dataset(generation_config_params, os.path.join(opt.output_dump, 'generator'), 'generation_dataset_processed.csv')
