import pandas as pd

from .estimator_argilla import ArgillaEstimator
from .estimator_llm import LLMEstimator
from .estimator_llm_batch import LLMBatchEstimator
from dataset.base_dataset import DatasetBase


class DummyEstimator:
    """
    A dummy callback for the Estimator class.
    This is a method to handle an empty estimator.
    """

    def __init__(self):
        self.cur_instruct = None
        print("📝 使用 DummyEstimator - 不會進行任何註釋操作")

    def calc_usage(self):
        """
        Dummy function to calculate the usage of the dummy estimator
        """
        return 0

    def apply(self, dataset: DatasetBase, batch_id: int, leq: bool = False):
        """
        Dummy function to mimic the apply method, returns an empty dataframe
        """
        print(f"📝 DummyEstimator.apply() - 跳過註釋步驟 (batch_id: {batch_id})")
        return pd.DataFrame()

    def init_chain(self, label_schema):
        """
        Dummy function to mimic the init_chain method
        """
        print(f"📝 DummyEstimator.init_chain() - 跳過初始化 (label_schema: {label_schema})")
        pass

def give_estimator(opt):
    # 檢查配置是否有效
    if not hasattr(opt, 'method') or not opt.method:
        print("⚠️  警告: 沒有指定 annotator 方法，使用 DummyEstimator")
        return DummyEstimator()
    
    try:
        if opt.method == 'argilla':
            return ArgillaEstimator(opt.config)
        elif opt.method == 'llm':
            return LLMEstimator(opt.config)
        elif opt.method == 'llm_batch':
            return LLMBatchEstimator(opt.config)
        else:
            print(f"⚠️  警告: 未知的 annotator 方法 '{opt.method}'，使用 DummyEstimator")
            return DummyEstimator()
    except Exception as e:
        print(f"⚠️  警告: 初始化 annotator 失敗 ({opt.method}): {e}")
        print("使用 DummyEstimator 作為備用")
        return DummyEstimator()
