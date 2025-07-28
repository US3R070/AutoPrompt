# Classifier評估機制示例

## 問題描述

假設我們有以下規定：
- 生成句子必須使用指定的成語
- 輸出不能超過15個字

## 問題場景

**User Input**: "請用成語'一針見血'造句，描述一個很長的句子，這個句子包含了非常多的詞彙和複雜的語法結構，用來測試模型的能力"

**Model Output**: "醫生一針見血地指出問題所在"

## 傳統評估方式的問題

如果將User Input和Model Output結合評估：
```
###User input:
請用成語'一針見血'造句，描述一個很長的句子，這個句子包含了非常多的詞彙和複雜的語法結構，用來測試模型的能力
####model prediction:
醫生一針見血地指出問題所在
```

Classifer可能會因為User Input超過15個字而判定為False，即使Model Output是符合規定的。

## 新的評估方式

### 1. 修改後的Classifier Prompt

```
Assistant is a large language model designed to classify whether generated text complies with given constraints.
Given a list of {batch_size} samples, evaluate whether the Model Output complies with the specified constraints.

### Task Instruction:
{task_instruction}

### Important Notes:
- User Input is provided for context only and should NOT be evaluated for compliance
- Only evaluate the Model Output for compliance with the specified constraints
- Focus your evaluation entirely on the Model Output section

### list of samples:
{samples}
```

### 2. 修改後的輸入格式

```
###User input (for context only):
請用成語'一針見血'造句，描述一個很長的句子，這個句子包含了非常多的詞彙和複雜的語法結構，用來測試模型的能力
####Model output (evaluate for compliance):
醫生一針見血地指出問題所在
```

### 3. Classifier評估結果

- **成語使用**: ✅ 使用了"一針見血"
- **字數限制**: ✅ 11個字，未超過15個字
- **結果**: True (符合規定)

## 關鍵改進

1. **明確分離**: 在prompt中明確指出User Input僅供上下文，不參與合規性評估
2. **專注評估**: Classifier只評估Model Output部分
3. **清晰標識**: 使用"(for context only)"和"(evaluate for compliance)"來明確區分
4. **避免干擾**: User Input的長度或其他特性不會影響Model Output的合規性評估

## 實際應用

這個改進特別適用於以下場景：
- User Input本身可能違反規定（如字數限制）
- 需要根據User Input的內容來評估Model Output的合規性
- 評估標準主要針對Model Output，但需要User Input作為上下文

## 配置修改

在`config/config_diff/config_classifier.yml`中：
```yaml
predictor:
    method: 'llm'
    config:
        prompt: 'prompts/predictor_completion/classifier_prediction.prompt'
```

在`generation_pipeline.py`中：
```python
classifier_df['text'] = classifier_df.apply(
    lambda row: f"###User input (for context only):\n{row['text']}\n####Model output (evaluate for compliance):\n{row['prediction']}", 
    axis=1
)
``` 