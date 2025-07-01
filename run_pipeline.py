from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, override_config
import argparse

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--basic_config_path', default='config/config_default.yml', type=str, help='Configuration file path')
parser.add_argument('--batch_config_path', default='',
                    type=str, help='Batch classification configuration file path')
parser.add_argument('--prompt',
                    default='你是一個回答者，你必須對輸入來產出一個最多10字的回答|--輸入 : 你口袋裡有多少錢輸出 : 62塊錢--輸入 : 你怎麼不去問問神奇海螺呢?輸出 : 我去問問神奇海螺',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='你是一個回答者，你必須用中文對應輸入，產出一個具體的、最多10字的回答',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--load_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=40, type=int, help='Number of iterations')

opt = parser.parse_args()

if opt.batch_config_path == '':
    # load the basic configuration using load_yaml
    config_params = load_yaml(opt.basic_config_path)
else:
    # override the basic configuration with the batch configuration
    config_params = override_config(opt.batch_config_path, config_file=opt.basic_config_path)

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt

# Initializing the pipeline
pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if (opt.load_path != ''):
    pipeline.load_state(opt.load_path)
best_prompt = pipeline.run_pipeline(opt.num_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_prompt['prompt'] + '\033[0m')

