import math
import time
from collections import OrderedDict

import numpy as np
import torch
from scipy.stats.stats import pearsonr, spearmanr
from torch import nn

from models.utils import replace_token
from trainers.utils import *

class Metric:
    def __init__(
        self,
        target_name=None,
        metric_name=None,
        PAD=None,
        STOP=None,
        prefix=None,
    ):
        super().__init__()
        self.reset()
        self.prefix = prefix
        self.target_name = target_name
        self.metric_name = metric_name
        self.PAD = PAD
        self.STOP = STOP

    def update(self, **kwargs):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def summarize(self, **kwargs):
        raise NotImplementedError

    def get_name(self):
        return self._prefix(self.metric_name)

    def _prefix_keys(self, summary):
        if self.prefix:
            summary = OrderedDict(
                {self._prefix(key): value for key, value in summary.items()}
            )
        return summary

    def _prefix(self, key):
        if self.prefix:
            return '{}_{}'.format(self.prefix, key)
        return key

    def token_mask(self, batch):
        target = self.get_target(batch)
        if self.PAD is not None:
            return target != self.PAD
        else:
            return torch.ones(
                target.shape, dtype=torch.uint8, device=target.device
            )

    def get_target(self, batch):
        target = getattr(batch, self.target_name)
        if self.STOP is not None:
            target = replace_token(target[:, 1:-1], self.STOP, self.PAD)
        return target

    def get_token_indices(self, batch):
        mask = self.token_mask(batch)
        return mask.view(-1).nonzero().squeeze()

    def get_predictions(self, model_out):
        predictions = model_out[self.target_name]
        return predictions

    def get_target_flat(self, batch):
        target_flat = self.get_target(batch).contiguous().view(-1)
        token_indices = self.get_token_indices(batch)
        return target_flat[token_indices]

    def get_predictions_flat(self, model_out, batch):
        predictions = self.get_predictions(model_out).contiguous()
        predictions_flat = predictions.view(-1, predictions.shape[-1]).squeeze()
        token_indices = self.get_token_indices(batch)
        return predictions_flat[token_indices]

    def get_tokens(self, batch):
        return self.token_mask(batch).sum().item()

class NLLMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(metric_name='NLL', **kwargs)

    def update(self, loss, batch, **kwargs):
        self.tokens += self.get_tokens(batch)
        self.nll += loss[self.target_name].item()

    def summarize(self):
        summary = {self.metric_name: self.nll / self.tokens}
        return self._prefix_keys(summary)

    def reset(self):
        self.nll = 0.0
        self.tokens = 0