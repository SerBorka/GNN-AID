from typing import Union, List, Callable

import sklearn
import torch


def asr(
        y_predict_clean,
        y_predict_after_attack_only,
        **kwargs
):
    if isinstance(y_predict_clean, torch.Tensor):
        if y_predict_clean.dim() > 1:
            y_predict_clean = y_predict_clean.argmax(dim=1)
        y_predict_clean.cpu()
    if isinstance(y_predict_after_attack_only, torch.Tensor):
        if y_predict_after_attack_only.dim() > 1:
            y_predict_after_attack_only = y_predict_after_attack_only.argmax(dim=1)
        y_predict_after_attack_only.cpu()
    return 1 - sklearn.metrics.accuracy_score(y_true=y_predict_clean, y_pred=y_predict_after_attack_only)


class AttackMetric:
    available_metrics = {
        'ASR': asr,
    }

    def __init__(
            self,
            name: str,
            **kwargs
    ):
        self.name = name
        self.kwargs = kwargs

    def compute(
            self,
            metrics_clean_model,
            metrics_after_attack
    ):
        if self.name in AttackMetric.available_metrics:
            return AttackMetric.available_metrics[self.name](
                metrics_clean_model,
                metrics_after_attack,
                **self.kwargs
            )
        raise NotImplementedError()



