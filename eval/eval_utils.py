from estimator.estimator_llm import LLMEstimator
import re


def set_function_from_iterrow(func):
    def wrapper(dataset):
        dataset['score'] = dataset.apply(func, axis=1)
        return dataset

    return wrapper


def set_ranking_function(params):
    def wrapper(dataset):
        # 直接比較 prediction 和 annotation
        # 將兩者都轉為字串以避免型別問題
        dataset['score'] = (dataset['prediction'].astype(str) == dataset['annotation'].astype(str)).astype(int)
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
