import numpy as np
import sklearn.metrics as fun

class MetricAbstract:
    def __init__(self):
        self.bigger= True # 指标越大越好，如果为False，表示越小越好
    def __str__(self):
        return self.__class__.__name__
    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class ACC(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.accuracy_score(y_true,y_pred)
class Pre(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.precision_score(y_true,y_pred, average='weighted')
class F1(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.f1_score(y_true,y_pred, average='weighted')
class Recall(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.recall_score(y_true,y_pred, average='weighted')
class Confusion(MetricAbstract):
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.confusion_matrix(y_true,y_pred)
class Hamming(MetricAbstract):
    def __init__(self):
        super().__init__()
        self.bigger = False
    def __call__(self, groundtruth, pred) -> float:
        y_true = groundtruth.astype(np.int64)
        y_pred = pred.astype(np.int64)
        assert y_pred.size == y_true.size
        return fun.hamming_loss(y_true,y_pred)

