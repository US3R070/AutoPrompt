#!/usr/bin/env python3
"""
分類任務的AutoPrompt優化管道
使用Ollama進行預測，OpenAI進行annotation和refinement
"""

import argparse
from pathlib import Path
from utils.config import load_yaml
import logging
import pandas as pd
import csv
from utils.reasoner import Reasoner
from utils.few_shot import FewShotSelector
from utils.llm_chain import get_llm
import dotenv
from classification_pipeline import ResOptimizationPipeline

dotenv.load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='運行分類任務的AutoPrompt優化管道')
    parser.add_argument('--config', type=str, default='config/config_classification.yml',
                       help='配置文件路徑')
    parser.add_argument('--prompt', type=str, 
                       default=
                       """請僅根據用戶訊息內容，判斷其是否有除自身健康狀況諮詢（例如症狀描述、個人診療、治療或用藥建議）以外的其他意圖。若訊息中出現任何與個人健康協助無關的問題或要求（如討論知識、經驗、健康資訊、政策流程、他人健康、分析原理、預防方式、或泛泛討論），請回傳「True」；若僅針對自身健康現狀尋求診斷、治療或用藥建議且無其他需求，請回傳「False」。只能從[\"True\", \"False\"]中選擇一者輸出，勿附加說明。\n/NO_THINK
                       """
                       
                       ,
                       help='初始prompt')
    parser.add_argument('--task_description', type=str,
                       default='準確判斷用戶訊息是否包含除健康要求外的其他意圖，有任何其他意圖的話回傳 "True"，僅健康意圖則回傳 "False"。不要回傳其他內容',
                       help='任務描述')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='優化步驟數')
    parser.add_argument('--output_dump', type=str, default='dump_classification',
                       help='輸出目錄')
    parser.add_argument('--load_path', type=str, default='',
                       help='加載檢查點路徑')
    parser.add_argument('--enable_few_shot', action='store_true', default=True,
                       help='是否啟用 few-shot 評估')
    parser.add_argument('--num_shots', type=int, default=2,
                       help='Few-shot 的範例數量')
    args = parser.parse_args()

    # 加載配置
    config = load_yaml(args.config)

    print(f"使用配置文件: {args.config}")
    print(f"初始prompt: {args.prompt}")
    print(f"任務描述: {args.task_description}")
    print(f"優化步驟數: {args.num_steps}")
    print(f"輸出目錄: {args.output_dump}")
    if args.enable_few_shot == True:
        print(f"Few-shot 啟用: {args.num_shots} shots")

    # 在創建優化管道之前，確保數據集包含必要欄位
    dataset_path = Path(config.dataset.records_path)
    if dataset_path.is_file():
        df = pd.read_csv(dataset_path)
        modified = False
        required_cols = {
            'id': lambda df: range(len(df)),
            'batch_id': 0,
            'prediction': pd.NA,
            'annotation': (df['answer'] if 'answer' in df.columns else pd.NA),
            'metadata': None,
            'score': pd.NA,
            'is_synthetic': False  # 預設 False
        }
        for col, default in required_cols.items():
            if col not in df.columns:
                if callable(default):
                    df.insert(0 if col == 'id' else len(df.columns), col, default(df))
                else:
                    df[col] = default
                modified = True
        # 你可以根據檔名或其他規則自動標記合成資料，例如：
        # if 'synthetic' in str(dataset_path).lower():
        #     df['is_synthetic'] = True
        # 也可以根據其他欄位自動標記（如有需要）
        if ('answer' in df.columns) and ('annotation' in df.columns) and df['annotation'].isna().all():
            df['annotation'] = df['answer'].astype(str)
        if modified:
            processed_dir = Path(args.output_dump)
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / 'classification_dataset_processed.csv'
            df['annotation'] = df['annotation'].astype(str)
            df.to_csv(processed_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
            config.dataset.records_path = str(processed_path)
            config.dataset.initial_dataset = str(processed_path)
            print(f"自動添加缺失欄位並寫入: {processed_path}")

    # 初始化 Reasoner
    llm = get_llm(config.reasoner.config.llm)
    analysis_prompt_template = (
        "你是一個專業的 prompt 工程師。以下是舊的 prompt 及其導致的錯誤案例，請分析這個 prompt 為什麼會導致這些錯誤，並給出具體的改進建議。\n\n"
        "舊 prompt:\n{old_prompt}\n\n錯誤案例:\n{error_cases}\n\n請用大約70字給出具體為甚麼會錯的根本原因。"
    )
    reasoner = Reasoner(llm, analysis_prompt_template)

    # 初始化 Few-shot Selector
    few_shot_selector = None
    if args.enable_few_shot:
        # 只用有正確答案的資料
        examples_pool = df[df['answer'].notna() & (df['answer'] != '')].copy()
        if len(examples_pool) >= args.num_shots:
            few_shot_selector = FewShotSelector(
                enable_few_shot=True,
                num_shots=args.num_shots,
                examples_pool=examples_pool,
                output_dir=args.output_dump
            )
            print(f"Few-shot 範例池包含 {len(examples_pool)} 個樣本")
        else:
            print(f"警告: 範例池樣本數 ({len(examples_pool)}) 少於所需的 shot 數 ({args.num_shots})，停用 few-shot")

    pipeline = ResOptimizationPipeline(
        config=config,
        task_description=args.task_description,
        initial_prompt=args.prompt,
        output_path=args.output_dump,
        reasoner=reasoner,
        few_shot_selector=few_shot_selector
    )

    print("[初始化] 先對現有資料集做一次 LLM 預測...")
    # 只對 is_synthetic=True 的資料做預測與 annotation
    synthetic_mask = pipeline.dataset.records['is_synthetic'] == True
    if synthetic_mask.any():
        pipeline.predictor.cur_instruct = pipeline.cur_prompt
        records = pipeline.predictor.apply(pipeline.dataset.records[synthetic_mask], pipeline.batch_id, leq=True)
        pipeline.dataset.records.loc[synthetic_mask, ['prediction']] = records['prediction'].values
        # 只對合成資料做 annotation（如果 annotation 欄位為空才補）
        missing_anno = pipeline.dataset.records.loc[synthetic_mask, 'annotation'].isna() | (pipeline.dataset.records.loc[synthetic_mask, 'annotation'] == '')
        if missing_anno.any():
            # 這裡可根據你的 annotation 流程補上 LLM annotation
            # pipeline.dataset.records.loc[synthetic_mask & missing_anno, 'annotation'] = ...
            pass  # 你可以在這裡插入自動標註邏輯
    batch_df = pipeline.dataset.get_leq(pipeline.batch_id)
    print("[初始化預測結果]")
    print(batch_df[['id', 'text', 'prediction', 'annotation']].head(10).to_string(index=False))

    processed_dataset_path = config.dataset.records_path
    current_df = pipeline.dataset.records
    current_df.to_csv(processed_dataset_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"已將初始化預測結果寫回 {processed_dataset_path}")
    
    print("開始運行AutoPrompt優化管道...")
    pipeline.run_pipeline(args.num_steps)
    best_result = pipeline.extract_best_prompt()
    print("\n" + "="*60)
    print("優化完成！")
    print(f"最佳prompt分數: {best_result['score']:.4f}")
    print("最佳prompt:")
    print("-" * 40)
    print(best_result['prompt'])
    print("-" * 40)

if __name__ == "__main__":
    main() 