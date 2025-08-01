from estimator.estimator_llm import LLMEstimator
import re


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def set_ranking_function(params):
    """
    Calculates a score based on the absolute difference between prediction and annotation.
    The score is normalized to a 0-1 range, where 1 is a perfect match.
    Assumes a 1-5 rating scale as per the project's ranking configuration.
    """
    def wrapper(dataset):
        import pandas as pd

        # Convert prediction and annotation to numeric, coercing errors to NaN
        pred = pd.to_numeric(dataset['prediction'], errors='coerce')
        anno = pd.to_numeric(dataset['annotation'], errors='coerce')

        # Define rating scale based on the project's configuration [1, 2, 3, 4, 5]
        min_rating = 1
        max_rating = 5
        rating_range = max_rating - min_rating

        # Calculate absolute difference
        abs_diff = (pred - anno).abs()

        # Calculate score using the formula: score = 1 - (diff / range)
        # A score of 1 is a perfect match, a score of 0 is the max possible distance.
        score = 1.0 - (abs_diff / rating_range)
        
        # Scores can be negative if prediction is outside the annotation range (e.g. pred=6, anno=1). Clip at 0.
        score = score.clip(lower=0)

        # Fill NaN values (from conversion errors or non-numeric labels) with 0, the lowest score.
        dataset['score'] = score.fillna(0)

        return dataset
    return wrapper


def set_classifier_then_ranker_function(classifier_params, ranker_params):
    """
    先用 classifier 檢查，再用 ranker 評分的評估函數
    
    Args:
        classifier_params: classifier 的配置參數
        ranker_params: ranker 的配置參數
    """
    # 初始化 classifier
    classifier = LLMEstimator(classifier_params)
    classifier.init_chain(classifier_params.label_schema)
    classifier.mode = 'prediction'
    
    # 初始化 ranker
    ranker = LLMEstimator(ranker_params)
    ranker.init_chain(ranker_params.label_schema)
    ranker.mode = 'score'
    
    def wrapper(dataset):
        result_dataset = dataset.copy()
        
        for index, row in result_dataset.iterrows():
            try:
                # 步驟1: 用 classifier 檢查是否符合要求
                classifier_input = row['text']
                classifier_response = classifier.llm.invoke(classifier_input)
                
                # 處理 classifier 回應
                if isinstance(classifier_response, dict) and 'text' in classifier_response:
                    classifier_text = classifier_response['text']
                else:
                    classifier_text = str(classifier_response)
                
                # 提取 True/False
                classifier_pattern = r'\b(True|False)\b'
                classifier_match = re.search(classifier_pattern, classifier_text, re.IGNORECASE)
                classifier_result = classifier_match.group(1) if classifier_match else 'False'
                
                # 步驟2: 根據 classifier 結果決定評分
                if classifier_result.lower() == 'true':
                    # 符合要求，用 ranker 評分
                    ranker_input = f"###User input:\n{row['text']}\n####model prediction:\n{row['prediction']}"
                    ranker_response = ranker.llm.invoke(ranker_input)
                    
                    # 處理 ranker 回應
                    if isinstance(ranker_response, dict) and 'text' in ranker_response:
                        ranker_text = ranker_response['text']
                    else:
                        ranker_text = str(ranker_response)
                    
                    # 提取 1-5 分數
                    ranker_pattern = r'\b([1-5])\b'
                    ranker_match = re.search(ranker_pattern, ranker_text)
                    score = int(ranker_match.group(1)) if ranker_match else 1
                    
                else:
                    # 不符合要求，直接給 1 分
                    score = 1
                
                result_dataset.at[index, 'score'] = score
                
            except Exception as e:
                result_dataset.at[index, 'score'] = 1
        
        return result_dataset
    
    return wrapper
