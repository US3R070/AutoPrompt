#!/usr/bin/env python3
"""
測試3步驟架構的腳本
"""

import os
import sys
from pathlib import Path

# 添加當前目錄到Python路徑
sys.path.append(str(Path(__file__).parent))

from run_generation_pipeline import *

def test_three_step_pipeline():
    """測試3步驟架構"""
    
    # 創建測試配置
    test_args = [
        '--classifier_config_path', 'config/config_diff/config_classifier.yml',
        '--ranker_config_path', 'config/config_diff/config_ranking.yml', 
        '--generation_config_path', 'config/config_diff/config_generation.yml',
        '--task_description', '你是一個短句回答者，對於每一種不同的短句，你有幾個模板回復可以選，選擇一個語意最適合的來回復對方',
        '--prompt', """你是一個短句回答者，你有幾個模板回復可以選，從模板中的name中選擇一個語意最適合的來回復對方
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
        '--output_dump', 'test_dump',
        '--num_classifier_steps', '3',
        '--num_ranker_steps', '3', 
        '--num_generation_steps', '3'
    ]
    
    # 模擬命令行參數
    sys.argv = ['test_script'] + test_args
    
    try:
        # 執行主程序
        opt = parser.parse_args()
        
        print("=" * 60)
        print("開始測試3步驟架構")
        print("=" * 60)
        
        # 載入三個配置
        classifier_config_params = override_config(opt.classifier_config_path)
        ranker_config_params = override_config(opt.ranker_config_path)
        generation_config_params = override_config(opt.generation_config_path)
        validate_generation_config(ranker_config_params, generation_config_params)
        
        task_description = opt.task_description
        initial_prompt = opt.prompt
        
        print(f"任務描述: {task_description}")
        print(f"初始提示詞: {initial_prompt}")
        
        # 步驟1: Classifier
        print("\n" + "=" * 50)
        print("步驟1: Classifier - 檢查硬性規定")
        print("=" * 50)
        
        if classifier_config_params.dataset.records_path != None:
            process_dataset(classifier_config_params, os.path.join(opt.output_dump, 'classifier'), 'classifier_dataset_processed.csv', type='classifier')
        
        classifier_pipeline = ResOptimizationPipeline(classifier_config_params, output_path=os.path.join(opt.output_dump, 'classifier'))
        
        if (classifier_pipeline.cur_prompt is None) or (classifier_pipeline.task_description is None):
            classifier_mod_prompt, classifier_mod_task_desc = modify_input_for_classifier(classifier_config_params, task_description, initial_prompt)
            classifier_pipeline.cur_prompt = classifier_mod_prompt
            classifier_pipeline.task_description = classifier_mod_task_desc
            print("Classifier 修改後的提示詞:", classifier_mod_prompt)
            print("Classifier 修改後的任務描述:", classifier_mod_task_desc)
        
        best_classifier_prompt = classifier_pipeline.run_pipeline(opt.num_classifier_steps)
        print("Classifier 最佳提示詞:", best_classifier_prompt)
        
        # 步驟2: Ranker
        print("\n" + "=" * 50)
        print("步驟2: Ranker - 語意評分")
        print("=" * 50)
        
        if ranker_config_params.dataset.records_path != None:
            process_dataset(ranker_config_params, os.path.join(opt.output_dump, 'ranker'), 'ranking_dataset_processed.csv', type='ranker')
        
        ranker_pipeline = RnkOptimizationPipeline(ranker_config_params, output_path=os.path.join(opt.output_dump, 'ranker'))
        
        if (ranker_pipeline.cur_prompt is None) or (ranker_pipeline.task_description is None):
            ranker_mod_prompt, ranker_mod_task_desc = modify_input_for_ranker(ranker_config_params, task_description, initial_prompt)
            ranker_pipeline.cur_prompt = ranker_mod_prompt
            ranker_pipeline.task_description = ranker_mod_task_desc
            print("Ranker 修改後的提示詞:", ranker_mod_prompt)
            print("Ranker 修改後的任務描述:", ranker_mod_task_desc)
        
        best_ranker_prompt = ranker_pipeline.run_pipeline(opt.num_ranker_steps)
        print("Ranker 最佳提示詞:", best_ranker_prompt)
        
        # 步驟3: Generation
        print("\n" + "=" * 50)
        print("步驟3: Generation - 生成結果")
        print("=" * 50)
        
        if generation_config_params.dataset.records_path != None:
            process_dataset(generation_config_params, os.path.join(opt.output_dump, 'generator'), 'generation_dataset_processed.csv', type='generator')
        
        # 設置 generation 的評估函數
        generation_config_params.eval.function_params = ranker_config_params.predictor.config
        generation_config_params.eval.function_params.instruction = best_ranker_prompt['prompt']
        generation_config_params.eval.function_params.label_schema = ranker_config_params.dataset.label_schema
        
        # 添加 classifier 的評估
        classifier_eval_config = classifier_config_params.predictor.config.copy()
        classifier_eval_config.instruction = best_classifier_prompt['prompt']
        classifier_eval_config.label_schema = classifier_config_params.dataset.label_schema
        
        generation_pipeline = GenOptimizationPipeline(generation_config_params, task_description, initial_prompt,
                                                   output_path=os.path.join(opt.output_dump, 'generator'),
                                                   classifier_eval_config=classifier_eval_config)
        
        best_generation_prompt = generation_pipeline.run_pipeline(opt.num_generation_steps)
        
        # 結果總結
        print('\n' + '\033[92m' + '=' * 60 + '\033[0m')
        print('\033[92m' + '最終結果總結:' + '\033[0m')
        print('\033[92m' + '=' * 60 + '\033[0m')
        print('\033[92m' + f'Classifier 最佳提示詞分數: {best_classifier_prompt["score"]:.4f}' + '\033[0m')
        print('\033[92m' + f'Ranker 最佳提示詞分數: {best_ranker_prompt["score"]:.4f}' + '\033[0m')
        print('\033[92m' + f'Generation 最佳提示詞分數: {best_generation_prompt["score"]:.4f}' + '\033[0m')
        print('\033[92m' + '測試完成！' + '\033[0m')
        
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_three_step_pipeline() 