import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from torch.distributions.normal import Normal

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from models.utils import apply_packed_sequence, make_loss_weights
from models import Estimator, Model
from data.utils import serialize_vocabs

@Model.register_subclass
class XLMREstimator(Estimator):

    def __init__(
        self, vocabs, opt, predictor_tgt=None, predictor_src=None, PreModelClass='XLMRPredictor'
        ):

        super().__init__(vocabs=vocabs, opt=opt, PreModelClass=PreModelClass)
        self.lstm_input_size = 768

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.opt.hidden_est,
            num_layers=self.opt.rnn_layers_est,
            batch_first=True,
            dropout=self.opt.dropout_est,
            bidirectional=True,
        )

        # for name, param in self.predictor_tgt.named_parameters():
        #     param.requires_grad = False

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def loss(self, model_out, batch):
        
        target, sentence_scores = batch
        sentence_pred = model_out['scores']
        sentence_scores = sentence_scores.to(self.device)

        if not self.sentence_sigma:
            loss = self.mse_loss(sentence_pred, sentence_scores)
        else:
            mean = model_out['sent_mu']
            sigma = model_out['sent_sigma']
            # Compute log-likelihood of x given mu, sigma
            normal = Normal(mean, sigma)
            # Renormalize on [0,1] for truncated Gaussian
            partition_function = (normal.cdf(1) - normal.cdf(0)).detach()
            nll = partition_function.log() - normal.log_prob(sentence_scores)
            loss = nll.sum()
        
        loss_dict = OrderedDict()
        loss_dict['loss'] = loss
        
        return loss_dict
    
    def get_mask(self, batch):
        """Compute Mask of Tokens for side.
        Args:
            batch: Namespace of tensors
            side: String identifier.
        """

        input_tensor, score = batch

        mask = torch.ones_like(input_tensor, dtype=torch.uint8)

        possible_padding = [0, 1, 2]
        for pad_id in possible_padding:

            mask &= torch.as_tensor(
                input_tensor != pad_id,
                device=mask.device,
                dtype=torch.uint8,
            )

        return mask
    
    def forward(self, batch):
    
        contexts_tgt, h_tgt = None, None
        contexts_src, h_src = None, None

        # Predict Target from Source
        model_out_tgt = self.predictor_tgt(batch)
        token_mask = self.get_mask(batch)
        target_lengths = token_mask.sum(1)

        contexts_tgt, h_tgt = apply_packed_sequence(
            self.lstm, model_out_tgt[0], target_lengths
        )

        sentence_input = self.make_sentence_input(h_tgt, h_src)
        outputs = self.predict_sentence(sentence_input)

        return outputs
    
    def save(self, path):
        vocabs = {}
        vocabs['target'] = Vocabs(self.vocabs)
        vocabs = serialize_vocabs(vocabs)
        model_dict = {
            'vocab': vocabs,
            self.__class__.__name__: {
                'state_dict': self.state_dict(),
            },
        }
        torch.save(model_dict, path)

class Vocabs:

    def __init__(self, vocabs):
        self.stoi = vocabs
        self.itos = {k:v for k,v in vocabs.items()}
