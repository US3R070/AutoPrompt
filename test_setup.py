#!/usr/bin/env python3
"""
測試AutoPrompt分類任務配置是否正確設置
"""

import os
import sys
import yaml
from pathlib import Path

def test_dependencies():
    """測試必要的依賴項是否已安裝"""
    print("🔍 檢查依賴項...")
    required_packages = [
        'langchain',
        'langchain_community', 
        'langchain_openai',
        'pandas',
        'easydict',
        'argilla'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - 未安裝")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少依賴項: {missing_packages}")
        print("請運行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有依賴項已安裝")
        return True

def test_config_files():
    """測試配置文件是否存在且格式正確"""
    print("\n🔍 檢查配置文件...")
    
    config_files = [
        'config/config_classification.yml',
        'config/llm_env.yml'
    ]
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"  ❌ {config_file} - 文件不存在")
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            print(f"  ✅ {config_file}")
        except yaml.YAMLError as e:
            print(f"  ❌ {config_file} - YAML格式錯誤: {e}")
            return False
    
    print("✅ 配置文件檢查通過")
    return True

def test_dataset():
    """測試數據集文件是否存在且格式正確"""
    print("\n🔍 檢查數據集文件...")
    
    dataset_file = 'dataset/classification_dataset.csv'
    if not Path(dataset_file).exists():
        print(f"  ❌ {dataset_file} - 文件不存在")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(dataset_file)
        
        if 'input' not in df.columns or 'answer' not in df.columns:
            print(f"  ❌ {dataset_file} - 缺少必要的欄位(input, answer)")
            return False
        
        print(f"  ✅ {dataset_file} - {len(df)} 條記錄")
        print(f"  ✅ 標籤分佈: {df['answer'].value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"  ❌ {dataset_file} - 讀取錯誤: {e}")
        return False

def test_prompt_files():
    """測試prompt文件是否存在"""
    print("\n🔍 檢查prompt文件...")
    
    prompt_files = [
        'prompts/predictor_completion/prediction.prompt',
        'prompts/predictor_completion/annotation.prompt',
        'prompts/predictor_completion/output_schemes.py'
    ]
    
    for prompt_file in prompt_files:
        if not Path(prompt_file).exists():
            print(f"  ❌ {prompt_file} - 文件不存在")
            return False
        print(f"  ✅ {prompt_file}")
    
    print("✅ Prompt文件檢查通過")
    return True

def test_ollama_connection():
    """測試Ollama連接"""
    print("\n🔍 檢查Ollama連接...")
    
    try:
        import requests
        response = requests.get('https://ollama.havenook.com/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"  ✅ Ollama連接正常 - 可用模型: {len(models)}")
            for model in models[:3]:  # 顯示前3個模型
                print(f"    - {model.get('name', 'Unknown')}")
            return True
        else:
            print(f"  ❌ Ollama連接失敗 - 狀態碼: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠️  無法連接到Ollama: {e}")
        print("    請確保Ollama服務正在運行: ollama serve")
        return False

def test_openai_config():
    """檢查OpenAI配置"""
    print("\n🔍 檢查OpenAI配置...")
    
    try:
        with open('config/llm_env.yml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        api_key = config.get('openai', {}).get('OPENAI_API_KEY', '')
        if not api_key or api_key == '':
            print("  ⚠️  OpenAI API密鑰未設置")
            print("    請在 config/llm_env.yml 中設置 OPENAI_API_KEY")
            return False
        else:
            print("  ✅ OpenAI API密鑰已設置")
            return True
    except Exception as e:
        print(f"  ❌ 讀取OpenAI配置失敗: {e}")
        return False

def main():
    print("🚀 AutoPrompt分類任務配置測試\n")
    
    tests = [
        ("依賴項檢查", test_dependencies),
        ("配置文件檢查", test_config_files),
        ("數據集檢查", test_dataset),
        ("Prompt文件檢查", test_prompt_files),
        ("Ollama連接檢查", test_ollama_connection),
        ("OpenAI配置檢查", test_openai_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} - 執行錯誤: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 測試結果總結:")
    print("="*50)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有測試通過！您可以開始使用分類任務了。")
        print("\n運行命令:")
        print("python run_classification_pipeline.py")
    else:
        print("\n⚠️  請修復上述問題後再運行分類任務。")
    
    return passed == total

if __name__ == "__main__":
    main() 