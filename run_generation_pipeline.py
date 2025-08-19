from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, modify_input_for_ranker, modify_input_for_classifier, validate_generation_config, override_config
import argparse
import os
from estimator.estimator_llm import LLMEstimator
import pandas as pd
import csv
from pathlib import Path
import numpy as np
from generation_pipeline import GenOptimizationPipeline
from ranker_pipeline import RnkOptimizationPipeline
from single_classify_pipeline import SingleClassifyOptimizationPipeline
from utils.llm_chain import MetaChain
from easydict import EasyDict


def process_dataset(config_params, output_dir, filename, type='ranker'):
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
            df['text'] = df.apply(lambda row: f"---\nInput:\n {row['text']}\n---\n{row['answer']}", axis=1)
            # df['text'] = (df['answer'] if 'answer' in df.columns else '')
            df = df[df['label'] != 1]
            df['annotation'] = (df['label'] if 'label' in df.columns else pd.NA)
            # 添加 is_synthetic 欄位：有 ground truth 資料的標記為 False（非合成資料）
            df['is_synthetic'] = False
            print(f"Ranker數據預處理完成：")
            print(f"  - 總數據量: {len(df)}")

        elif type == 'generator':
            df['text'] = (df['text'] if 'text' in df.columns else '')
            df['score'] = (df['label'] if 'label' in df.columns else pd.NA)
            # 添加 is_synthetic 欄位：有 ground truth 資料的標記為 False（非合成資料）
            df['is_synthetic'] = False
        elif type == 'classifier':
            # Classifier特殊預處理：將label為1分的標記為False，其他標記為True
            df['text'] = df.apply(lambda row: f"---\nInput:\n {row['text']}\n---Model Output:\n{row['answer']}", axis=1)
            
            # 預處理label：1分 -> False，其他 -> True
            if 'label' in df.columns:
                df['annotation'] = df['label'].apply(lambda x: 'False' if str(x).strip() == '1' else 'True')
                # 添加 is_synthetic 欄位：有 ground truth 資料的標記為 False（非合成資料）
                df['is_synthetic'] = False
                print(f"Classifier數據預處理完成：")
                print(f"  - 總數據量: {len(df)}")
                print(f"  - 標記為False (原label=1): {len(df[df['annotation'] == 'False'])}")
                print(f"  - 標記為True (原label!=1): {len(df[df['annotation'] == 'True'])}")
                print(f"  - 有ground truth資料: {len(df)} (全部標記為非合成資料)")
            else:
                df['annotation'] = pd.NA
                # 沒有 ground truth 資料的標記為 True（合成資料）
                df['is_synthetic'] = True

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
parser.add_argument('--classifier_config_path', default='config/config_diff/config_classifier.yml', type=str, help='Configuration file path')

parser.add_argument('--task_description',
                    default='你是一個回答者，對於每一種不同的輸入，你有幾個模板回復可以選，選擇一個語意最適合的來回復對方',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--prompt',
                    default=
                    """你是一個回答者，你有幾個模板回復可以選，從模板中的name中選擇一個語意最適合的來回復對方
                    注意，目的在回復消息，不是重複短句的內容
                    ---
                    短句 : 誰會想在凌晨3點吃美味蟹堡
                    模板 : {"name":"誰會想在凌晨3點吃美味蟹堡","number":"SS0001"}{"name":"你為什麼不問問神奇海螺呢","number":"SS0001"}...
                    回答 : 你為什麼不問問神奇海螺呢
                    ---
                    回答選模板的第2個選項的name(你為什麼不問問神奇海螺呢)，這個選項較可以回復對方的問題
                    請注意，這個選項的number:"SS0001"並不是數字，而是模板回復的編號，請不要使用這個編號來回復對方
                    因此並不是找最相近的，而是語義可以對的上的
                    """,
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--load_dump', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_classifier_steps', default=1, type=int, help='Number of iterations for classifier')
parser.add_argument('--num_ranker_steps', default=1, type=int, help='Number of iterations for ranker')
parser.add_argument('--num_generation_steps', default=5, type=int, help='Number of iterations for generation')
parser.add_argument('--has_initial_data', action='store_true', help='資料集是否有初始標註資料（有則 batch_id==0 不做 annotation）')

opt = parser.parse_args()

# 載入三個配置
classifier_config_params = override_config(opt.classifier_config_path)
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

# 步驟1: Classifier - 檢查硬性規定
print("=" * 50)
print("步驟1: Classifier - 檢查硬性規定")
print("=" * 50)

# 處理 classifier 資料集
if classifier_config_params.dataset.records_path != None:
    process_dataset(classifier_config_params, os.path.join(opt.output_dump, 'classifier'), 'classifier_dataset_processed.csv', type='classifier')

# 創建 meta_chain 並傳入 SingleClassifyOptimizationPipeline
meta_chain = MetaChain(classifier_config_params)

classifier_pipeline = SingleClassifyOptimizationPipeline(classifier_config_params, output_path=os.path.join(opt.output_dump, 'classifier'), meta_chain=meta_chain)
if opt.load_dump != '':
    classifier_pipeline.load_state(os.path.join(opt.load_dump, 'classifier'))
    classifier_pipeline.predictor.init_chain(classifier_config_params.dataset.label_schema)

if (classifier_pipeline.cur_prompt is None) or (classifier_pipeline.task_description is None):
    classifier_mod_prompt, classifier_mod_task_desc = modify_input_for_classifier(classifier_config_params, opt.task_description, opt.prompt)
    classifier_pipeline.cur_prompt = classifier_mod_prompt
    classifier_pipeline.task_description = classifier_mod_task_desc
    print("classifier_mod_prompt : ", classifier_mod_prompt)
    print("classifier_mod_task_desc : ", classifier_mod_task_desc)

best_classifier_prompt = classifier_pipeline.run_pipeline(opt.num_classifier_steps)
print("best_prompt for classifier : ", best_classifier_prompt)

# 步驟2: Ranker - 語意評分
print("=" * 50)
print("步驟2: Ranker - 語意評分")
print("=" * 50)

ranker_config_params.eval.function_params = ranker_config_params.annotator.config
ranker_config_params.eval.function_params.label_schema = ranker_config_params.dataset.label_schema

# 處理 ranking 資料集
if ranker_config_params.dataset.records_path != None:
    process_dataset(ranker_config_params, os.path.join(opt.output_dump, 'ranker'), 'ranking_dataset_processed.csv', type='ranker')

ranker_pipeline = RnkOptimizationPipeline(ranker_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
if opt.load_dump != '':
    ranker_pipeline.load_state(os.path.join(opt.load_dump, 'ranker'))
    ranker_pipeline.predictor.init_chain(ranker_config_params.dataset.label_schema)

if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
    ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(ranker_config_params, opt.task_description, opt.prompt)
    ranker_pipeline.cur_prompt = ranker_mod_prompt
    ranker_pipeline.task_description = ranker_mod_task_desc
    print("ranker_mod_prompt : ", ranker_mod_prompt)
    print("ranker_mod_task_desc : ", ranker_mod_task_desc)

best_ranker_prompt = ranker_pipeline.run_pipeline(opt.num_ranker_steps)
print("best_prompt for ranker : ", best_ranker_prompt)

# 步驟3: Generation - 生成結果
print("=" * 50)
print("步驟3: Generation - 生成結果")
print("=" * 50)

# --- 新增：從label=5的資料創建並儲存標準輸出模板 ---
def create_and_save_golden_template(dataset_path, output_dir):
    """
    從指定資料集中篩選 label 為 5 的資料，
    提取其 'answer' 作為黃金模板，並儲存至檔案。
    同時返回該模板內容供後續流程使用。
    """
    # 根據使用者要求，模板檔案直接儲存在 dump/ 底下
    template_path = Path(output_dir) / 'generation_output_template.txt'
    
    if not Path(dataset_path).is_file():
        print(f"資料集檔案不存在於 {dataset_path}，無法建立標準模板。")
        return "" # 返回空字串，避免後續流程出錯
    
    try:
        df = pd.read_csv(dataset_path)
        if 'label' not in df.columns or 'answer' not in df.columns:
            print("資料集必須包含 'label' 和 'answer' 欄位才能建立標準模板。")
            return ""

        # 篩選 label 為 '5' 的資料 (轉換為字串以確保比對正確)
        golden_samples = df[df['label'].astype(str).str.strip() == '5']
        
        if golden_samples.empty:
            print("在資料集中找不到 label 為 '5' 的樣本，無法建立標準模板。")
            return ""

        # 提取第一筆符合條件的 'answer' 作為模板
        golden_template = golden_samples['answer'].iloc[0]
        
        # 建立目錄並寫入檔案
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(golden_template)
            
        print(f"已成功從 label=5 的資料建立標準輸出模板，並儲存至: {template_path}")
        
        # 返回模板內容，供 generation pipeline 用於 error analysis
        return golden_template

    except Exception as e:
        print(f"建立標準模板時發生錯誤: {e}")
        return ""

# 使用 decider.csv 的 answer 來生成格式定義
# output_dir 直接使用 opt.output_dump (即 'dump/')
decider_csv_path = 'dataset/decider.csv'
output_format_definition = create_and_save_golden_template(
    dataset_path=decider_csv_path,
    output_dir=opt.output_dump
)
# --- 標準模板創建結束 ---

# 處理 generation 資料集
if generation_config_params.dataset.records_path != None:
    process_dataset(generation_config_params, os.path.join(opt.output_dump, 'generator'), 'generation_dataset_processed.csv', type='generator')

# 為了讓 OptimizationPipeline 中的 Eval class 能夠成功初始化，我們需要保留 function_params 的設定
generation_config_params.eval.function_params = ranker_config_params.predictor.config
generation_config_params.eval.function_params.instruction = best_ranker_prompt['prompt']
generation_config_params.eval.function_params.label_schema = ranker_config_params.dataset.label_schema

generation_pipeline = GenOptimizationPipeline(
    config=generation_config_params, 
    task_description=task_description, 
    initial_prompt=initial_prompt,
    output_path=os.path.join(opt.output_dump, 'generator'),
    classifier_config=classifier_config_params,
    ranker_config=ranker_config_params,
    best_classifier_prompt=best_classifier_prompt,
    best_ranker_prompt=best_ranker_prompt,
    output_format_definition=output_format_definition
)

if opt.load_dump != '':
    generation_pipeline.load_state(os.path.join(opt.load_dump, 'generator'))

best_generation_prompt = generation_pipeline.run_pipeline(opt.num_generation_steps)

# 新增：為最佳提示詞添加 few-shot 範例，並將其儲存起來
few_shot_block = generation_pipeline.get_few_shot_examples(max_examples=generation_config_params.few_shot_examples)
if few_shot_block:
    best_generation_prompt['prompt'] += "\n\n" + few_shot_block 
    best_generation_prompt['few_shot_examples'] = few_shot_block

print('\033[92m' + 'Calibrated prompt score:', str(best_generation_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_generation_prompt['prompt'] + '\033[0m')

print('\033[92m' + '=' * 50 + '\033[0m')
print('\033[92m' + '最終結果總結:' + '\033[0m')
print('\033[92m' + '=' * 50 + '\033[0m')

print('\033[92m' + f'Generation 最佳提示詞分數: {best_generation_prompt["score"]:.4f}' + '\033[0m')

# Final validation
print('\033[92m' + '=' * 50 + '\033[0m')
print('\033[92m' + 'Final Validation' + '\033[0m')
print('\033[92m' + '=' * 50 + '\033[0m')

from tqdm import tqdm
from estimator import give_estimator
from estimator.estimator_llm import LLMEstimator
from langchain_core.prompts import PromptTemplate
import numpy as np
import random
import pandas as pd

def validate_and_compare(initial_prompt, best_prompt, dataset, llm, classifier_predictor, ranker_predictor, num_examples=5):
    
    # --- 1. Generation Step ---
    template = PromptTemplate.from_template("{prompt}\n{user_input}")
    generation_chain = template | llm
    
    generated_data = []
    for record in tqdm(dataset, desc="Step 1/3: Generating Predictions"):
        initial_text = generation_chain.invoke({"prompt": initial_prompt, "user_input": record['text']}).content
        best_text = generation_chain.invoke({"prompt": best_prompt, "user_input": record['text']}).content
        generated_data.append({
            "user_input": record['text'],
            "initial_output": initial_text,
            "best_output": best_text
        })

    # --- 2. Classification Step ---
    # Prepare DataFrames for classification
    initial_classifier_df = pd.DataFrame([{
        'id': i,
        'text': f"---\nInput:\n {data['user_input']}\n---Model Output:\n{data['initial_output']}"
    } for i, data in enumerate(generated_data)])
    
    best_classifier_df = pd.DataFrame([{
        'id': i,
        'text': f"---\nInput:\n {data['user_input']}\n---Model Output:\n{data['best_output']}"
    } for i, data in enumerate(generated_data)])

    # Run classification with a single progress bar
    with tqdm(total=2, desc="Step 2/3: Classifying") as pbar:
        initial_class_results = classifier_predictor.apply_dataframe(initial_classifier_df)
        pbar.update(1)
        best_class_results = classifier_predictor.apply_dataframe(best_classifier_df)
        pbar.update(1)

    # --- 3. Ranking Step ---
    initial_scores = np.ones(len(generated_data)) # Default score is 1 (penalty)
    best_scores = np.ones(len(generated_data))    # Default score is 1 (penalty)

    # Prepare DataFrames for ranking (only for those that passed classification)
    initial_ranker_df = initial_class_results[initial_class_results['prediction'].str.lower() == 'true'].copy()
    best_ranker_df = best_class_results[best_class_results['prediction'].str.lower() == 'true'].copy()

    # Run ranking with a single progress bar
    if not initial_ranker_df.empty or not best_ranker_df.empty:
        with tqdm(total=(not initial_ranker_df.empty) + (not best_ranker_df.empty), desc="Step 3/3: Ranking") as pbar:
            if not initial_ranker_df.empty:
                initial_rank_results = ranker_predictor.apply_dataframe(initial_ranker_df)
                # Update scores for the ones that were ranked
                for _, row in initial_rank_results.iterrows():
                    initial_scores[row['id']] = float(row['prediction'])
                pbar.update(1)
            
            if not best_ranker_df.empty:
                best_rank_results = ranker_predictor.apply_dataframe(best_ranker_df)
                # Update scores for the ones that were ranked
                for _, row in best_rank_results.iterrows():
                    best_scores[row['id']] = float(row['prediction'])
                pbar.update(1)

    # --- 4. Finalization ---
    # Select random examples for comparison
    examples = []
    if len(generated_data) > num_examples:
        example_indices = random.sample(range(len(generated_data)), num_examples)
    else:
        example_indices = range(len(generated_data))
    
    for i in example_indices:
        examples.append(generated_data[i])

    return np.mean(initial_scores), np.mean(best_scores), examples




# Load dataset
validation_dataset = generation_pipeline.dataset.records.to_dict('records')

# LLM for generation
# To replicate the generation step, we must use the exact same configuration 
# as the predictor within the pipeline. The correct config object is under 'predictor.config'.
llm_predictor_config = generation_config_params.predictor.config

# The 'mode' attribute is not in the YAML but is required by the LLMEstimator.
# We'll add it manually to match the likely behavior of the pipeline.
llm_predictor_config.mode = 'prediction'

# Now, create the estimator with the correct config.
llm_estimator = LLMEstimator(opt=llm_predictor_config)
# Initialize the chain to load the prompt and the LLM.
llm_estimator.init_chain(label_schema=generation_config_params.dataset.label_schema)
# Extract the llm instance for raw invocation.
llm = llm_estimator.chain.llm

# Classifier Predictor
# We need the full predictor config object, which includes the 'method'.
classifier_predictor_obj = classifier_config_params.predictor
# We then inject the best prompt and few-shot examples into the inner 'config' object.
classifier_predictor_obj.config.instruction = best_classifier_prompt['prompt']
if 'few_shot_examples' in best_classifier_prompt:
    classifier_predictor_obj.config.few_shot_examples = best_classifier_prompt['few_shot_examples']
# Now, give_estimator receives the correct object with the 'method' attribute.
classifier_predictor = give_estimator(classifier_predictor_obj)
classifier_predictor.init_chain(label_schema=classifier_config_params.dataset.label_schema)

# Ranker Predictor
# We apply the same logic as for the classifier predictor.
ranker_predictor_obj = ranker_config_params.predictor
ranker_predictor_obj.config.instruction = best_ranker_prompt['prompt']
if 'few_shot_examples' in best_ranker_prompt:
    ranker_predictor_obj.config.few_shot_examples = best_ranker_prompt['few_shot_examples']
ranker_predictor = give_estimator(ranker_predictor_obj)
ranker_predictor.init_chain(label_schema=ranker_config_params.dataset.label_schema)

# Construct the final best prompt with few-shot examples, just like in the training loop.
final_best_prompt = best_generation_prompt['prompt']

# --- Run Validation and Print Results ---
initial_score, best_score, comparison_examples = validate_and_compare(
    initial_prompt,
    final_best_prompt,
    validation_dataset,
    llm,
    classifier_predictor,
    ranker_predictor
)

print(f"\n--- Final Scores ---")
print(f"Initial Prompt Score: {initial_score:.4f}")
print(f"Best Prompt Score: {best_score:.4f}")
print(f"Score Improvement (Best - Initial): {best_score - initial_score:.4f}")

print("\n--- Comparison Examples ---")
for i, example in enumerate(comparison_examples):
    print(f"--- Example {i+1} ---")
    print(f"User Input: {example['user_input']}")
    print(f"Output (Initial Prompt): {example['initial_output']}")
    print(f"Output (Best Prompt): {example['best_output']}\n")

