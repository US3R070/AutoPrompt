# AutoPrompt 分類任務實現

這是一個基於AutoPrompt框架的分類任務實現，專門用於優化分類prompt。該實現使用遠程Ollama進行預測，OpenAI API進行annotation和refinement。

## 功能特性

- 🎯 **分類任務優化**: 專門針對情感分析等分類任務設計
- 🦙 **Ollama支持**: 使用本地或遠程Ollama模型進行預測評估
- 🤖 **OpenAI集成**: 利用OpenAI API進行標註和prompt改進
- 📊 **準確率評估**: 使用簡單的準確率指標評估prompt性能
- 📈 **自動優化**: 自動生成邊界案例並改進prompt

## 文件結構

```
dataset/
├── classification_dataset.csv        # 示例分類數據集
config/
├── config_classification.yml         # 分類任務配置文件
├── llm_env.yml                      # LLM環境配置(已更新支持Ollama)
prompts/predictor_completion/
├── annotation.prompt                # 標註prompt模板
├── output_schemes.py               # 輸出解析器(已更新)
utils/
├── config.py                       # 配置工具(已添加Ollama支持)
run_classification_pipeline.py       # 分類任務運行腳本
```

## 安裝依賴

確保安裝了所需的依賴包：

```bash
pip install langchain-community
pip install langchain-openai
```

## 配置設置

### 1. 設置OpenAI API密鑰

編輯 `config/llm_env.yml`：

```yaml
openai:
  OPENAI_API_KEY: 'your-openai-api-key-here'
```

### 2. 確保Ollama運行

確保Ollama服務正在運行並可訪問：

```bash
# 啟動Ollama服務 (如果使用本地)
ollama serve

# 測試模型可用性
ollama list
```

如果使用遠程Ollama，請在配置文件中更新`base_url`。

## 使用方法

### 基本使用

```bash
python run_classification_pipeline.py
```

### 自定義參數

```bash
python run_classification_pipeline.py \
    --prompt "請分析這個文本的情感傾向，回答positive或negative" \
    --task_description "Assistant是專業的情感分析器" \
    --num_steps 20 \
    --output_dump "my_classification_results"
```

### 參數說明

- `--config`: 配置文件路徑 (默認: `config/config_classification.yml`)
- `--prompt`: 初始prompt
- `--task_description`: 任務描述
- `--num_steps`: 優化步驟數 (默認: 15)
- `--output_dump`: 輸出目錄 (默認: `dump_classification`)
- `--load_path`: 從檢查點恢復的路徑

## 配置說明

### 核心配置 (`config/config_classification.yml`)

```yaml
# 數據集配置
dataset:
    label_schema: ["Yes", "No"]    # 分類標籤
    max_samples: 30                          # 最大樣本數

# 預測器配置 (使用Ollama)
predictor:
    method: 'llm'
    config:
        llm:
            type: 'Ollama'
            name: 'llama3'                   # Ollama模型名稱
            base_url: 'http://localhost:11434'  # Ollama服務地址

# 標註器配置 (使用OpenAI)
annotator:
    method: 'llm'
    config:
        llm:
            type: 'OpenAI'
            name: 'gpt-3.5-turbo'

# Meta-prompt配置 (使用OpenAI)
llm:
    name: 'gpt-4'
    type: 'OpenAI'
```

## 數據集格式

CSV文件應包含兩個欄位：

```csv
input,answer
這部電影非常精彩，我給滿分,positive
電影很無聊，浪費時間,negative
演員表演很棒，劇情也很有趣,positive
```

## 輸出結果

運行完成後，您將獲得：

1. **優化後的prompt**: 經過多輪改進的最佳prompt
2. **性能分數**: 在測試集上的準確率
3. **詳細日誌**: 保存在輸出目錄中的優化過程記錄
4. **檢查點文件**: 可用於恢復優化過程

## 故障排除

### Ollama連接問題

```bash
# 檢查Ollama服務狀態
curl http://localhost:11434/api/tags

# 拉取所需模型
ollama pull llama3
```

### OpenAI API問題

- 確保API密鑰正確設置
- 檢查API額度是否充足
- 確認網絡連接正常

### 模型輸出格式問題

如果遇到解析錯誤，檢查：
- 模型輸出是否遵循指定格式
- prompt模板是否正確
- 標籤名稱是否匹配配置

## 自定義擴展

### 添加新的分類標籤

1. 更新 `config/config_classification.yml` 中的 `label_schema`
2. 更新數據集文件中的標籤
3. 如需要，調整prompt模板

### 使用不同的Ollama模型

在配置文件中更改 `predictor.config.llm.name` 為您想要的模型名稱：

```yaml
predictor:
    config:
        llm:
            name: 'llama3:8b'  # 或其他可用模型
```

## 注意事項

- 確保Ollama模型已下載並可用
- OpenAI API調用會產生費用，請注意預算設置
- 建議先用較少的步驟測試，確認配置正確後再進行完整優化
- 保存輸出目錄，以便後續分析和恢復

## 效果預期

使用這個框架，您可以期望：

- 🎯 自動生成更準確的分類prompt
- 📊 獲得量化的性能改進指標
- 🔄 通過迭代優化持續改進prompt質量
- 💰 相比純OpenAI方案，顯著降低推理成本 