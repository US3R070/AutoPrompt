import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import eval.eval_utils as utils

class Eval:
    """
    The Eval class is responsible to calculate the score and the large errors
    """

    def __init__(self, config, analyzer=None, label_schema=None):
        """
        Initialize a new instance of the Eval class.
        :param config: The configuration file (EasyDict)
        :analyzer (optional): A chain that analyze the errors
        :label_schema (optional): The label schema
        """
        self.score_function_name = config.function_name
        self.score_func = self.get_eval_function(config)
        self.function_params = config.get('function_params')  # 新增此行
        self.num_errors = config.num_large_errors
        self.error_threshold = config.error_threshold
        self.dataset = None
        self.mean_score = None
        self.label_schema = label_schema
        self.errors = None
        self.history = []
        self.analyzer = analyzer

    @staticmethod
    def get_eval_function(config: dict):
        """
        Returns the eval function
        :param config: The eval configuration
        :return: The function implementation on a record
        """
        if config.function_name == 'accuracy':
            return utils.set_function_from_iterrow(lambda record: record['annotation'] == record['prediction'])
        elif config.function_name == 'ranking':
            return utils.set_ranking_function(config.function_params)
        elif config.function_name == 'classifier_then_ranker':
            # 需要 classifier_params 和 ranker_params
            if hasattr(config, 'classifier_params') and hasattr(config, 'ranker_params'):
                return utils.set_classifier_then_ranker_function(config.classifier_params, config.ranker_params)
            else:
                raise ValueError("classifier_then_ranker 需要 classifier_params 和 ranker_params")
        else:
            raise NotImplementedError("Eval function not implemented")

    def eval_score(self) -> float:
        """
        Calculate the score on each row and return the mean score.
        :return: The mean score
        """
        # filter out the discarded samples
        self.dataset = self.dataset[(self.dataset['prediction'] != 'Discarded') &
                                    (self.dataset['annotation'] != 'Discarded')]
        self.dataset = self.score_func(self.dataset)
        # print("self.dataset['score'] : ",self.dataset['score'])
        self.mean_score = self.dataset['score'].mean()
        # print("self.mean_score : ",self.mean_score)
        return self.mean_score

    def get_max_score(self, warmup=0):
        """
        Return the maximum 'mean score' (with respect to all history epochs, starting form warmup, up to last) and the epoch index of the maximum score
        :return: The epoch index of the maximum score, and the maximum score
        """
        max_idx = np.argmax([epoch['score'] for epoch in self.history[warmup:-1]])
        max_idx += warmup
        return max_idx, self.history[max_idx]['score']


    def large_error_to_str(self, error_df: pd.DataFrame, num_large_errors_per_label: int) -> str:
        """
        Return a string that contains the large errors
        :param error_df: A dataframe contains all the mislabeled samples
        :param num_large_errors_per_label: The (maximum) number of large errors per label
        :return: A string that contains the large errors that is used in the meta-prompt
        """
        required_columns = ['annotation', 'text', 'score', 'prediction']
        
        # 添加調試信息
        print(f"🔍 large_error_to_str 調試信息:")
        print(f"  - 總錯誤數量: {len(error_df)}")
        print(f"  - 錯誤 DataFrame 欄位: {error_df.columns.tolist()}")
        if len(error_df) > 0:
            print(f"  - 錯誤樣本:")
            print(error_df)
        
        label_schema = error_df['annotation'].unique()
        if self.score_function_name == 'ranking':
            gt_name = 'Rank'
        else:
            gt_name = 'GT'
        error_res_df_list = []
        txt_res = ''
        for label in label_schema:
            cur_df = error_df[error_df['annotation'] == label]
            print(f"  - Label '{label}' 的錯誤數量: {len(cur_df)}")
            
            # 如果錯誤數量超過限制，隨機採樣
            if len(cur_df) > num_large_errors_per_label:
                cur_df = cur_df.sample(frac=1.0, random_state=42)[:num_large_errors_per_label]

            
            if len(cur_df) > 0:
                error_res_df_list.append(cur_df[required_columns])
        
        if len(error_res_df_list) > 0:
            error_res_df = pd.concat(error_res_df_list, ignore_index=True)
            error_res_df = error_res_df.sample(frac=1.0, random_state=42)
            for i, row in error_res_df.iterrows():
                txt_res += f"Sample: {row.text}\nPrediction: {row.prediction}, {gt_name}: {row.annotation}\n#\n"
        return txt_res

    def sample_to_text(self, sample: dict, num_errors_per_label: int = 0, is_score: bool = True) -> str:
        """
        Return a string that organize the information of from the step run for the meta-prompt
        :param sample: The eval information for specific step
        :param num_errors_per_label: The max number of large errors per class that will appear in the meta-prompt
        :param is_score: If True, add the score information to the meta-prompt
        :return: A string that contains the information of the step run
        """
        if is_score:
            return f"####\n##Prompt Score: {sample['score']:.2f}\n##Prompt:\n{sample['prompt']}\n#################\n"
        else:
            return f"####\n##Prompt:\n{sample['prompt']}\n{self.large_error_to_str(sample['errors'], num_errors_per_label)}####\n "

    def add_history(self, prompt: str, task_description: str):
        """
        Add the current step information to the history
        :param prompt: The current prompt
        :param task_description: The task description
        """
        conf_matrix = None
        large_error_to_str = self.large_error_to_str(self.errors, self.num_errors)
        prompt_input = {'task_description': task_description, 'accuracy': self.mean_score, 'prompt': prompt,
                                         'failure_cases': large_error_to_str}
        if self.score_function_name == 'accuracy':
            # conf_matrix = confusion_matrix(self.dataset['annotation'],
            #                                self.dataset['prediction'], labels=self.label_schema)
            
            # print prompt_input
            print(prompt+"\n")
            
            for i in range(len(self.dataset)):
                print(f"{self.dataset['annotation'][i]} {self.dataset['prediction'][i]}")
            conf_matrix = confusion_matrix(self.dataset['annotation'],
                                           self.dataset['prediction'])
            conf_text = f"Confusion matrix columns:{self.label_schema} the matrix data:"
            for i, row in enumerate(conf_matrix):
                conf_text += f"\n{self.label_schema[i]}: {row}"
            prompt_input['confusion_matrix'] = conf_text
        elif self.score_function_name == 'ranking':
            prompt_input['labels'] = self.label_schema
            prompt_input['confusion_matrix'] = "N/A for ranking tasks."
            print('prompt_input : ',prompt_input)
        analysis = self.analyzer.invoke(prompt_input)

        self.history.append({'prompt': prompt, 'score': self.mean_score,
                             'errors': self.errors, 'confusion_matrix': conf_matrix, 'analysis': analysis['text']})

    def extract_errors(self) -> pd.DataFrame:
        """
        Extract the errors from the dataset
        :return: records that contains the errors
        """
        df = self.dataset
        print(f"🔍 extract_errors 調試信息:")
        print(f"  - 總樣本數量: {len(df)}")
        print(f"  - 樣本分數分布: {df['score']}")
        print(f"  - 錯誤閾值: {self.error_threshold}")
        
        # 提取錯誤樣本（分數小於閾值的樣本）
        err_df = df[df['score'] < self.error_threshold]
        print(f"  - 錯誤樣本數量: {len(err_df)}")
        
        if len(err_df) > 0:
            print(f"  - 錯誤樣本的 annotation 分布: {err_df['annotation'].value_counts().to_dict()}")
            print(f"  - 錯誤樣本的 prediction 分布: {err_df['prediction'].value_counts().to_dict()}")
        
        err_df = err_df.sort_values(by=['score'])
        self.errors = err_df
        return self.errors

    def extract_correct(self) -> pd.DataFrame:
        """
        Extract the correct samples from the dataset
        :return: records that contains the correct samples
        """
        df = self.dataset
        return df[df['score'] > self.error_threshold]

    def extract_boundary_predictions(self) -> pd.DataFrame:
        """
        Extract boundary samples on which the model is uncertain
        :return: records that contains boundary samples
        """
        pass