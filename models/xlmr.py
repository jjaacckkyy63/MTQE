import torch
from torch import nn
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, XLMRobertaModel
from models.utils import apply_packed_sequence, make_loss_weights
from models import Model
from data.utils import deserialize_vocabs

@Model.register_subclass
class XLMRPredictor(Model):
    
    def __init__(self, vocabs, opt, predict_inverse=False):
        super(XLMRPredictor, self).__init__(vocabs=vocabs, opt=opt)
        
        self.xlmr = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.model = nn.Sequential(*list(self.xlmr.children())[:-1])
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def loss(self, model_out, batch, target_side=None):
        if not target_side:
            target_side = self.target_side
        target = getattr(batch, target_side)
        # There are no predictions for first/last element
        target = replace_token(target[:, 1:-1], self.opt.STOP_ID, self.opt.PAD_ID)
        # Predicted Class must be in dim 1 for xentropyloss
        logits = model_out[target_side]
        logits = logits.transpose(1, 2)
        loss = self._loss(logits, target)
        loss_dict = OrderedDict()
        loss_dict[target_side] = loss
        loss_dict['loss'] = loss
        return loss_dict

    def forward(self, batch):
        target, score = batch
        target = target.to(self.device)
        source_outputs = self.xlmr(target)
        return source_outputs
    
    @classmethod
    def from_dict(cls, model_dict, opt, PreModelClass=None, vocabs=None):
        if not vocabs:
            vocabs = deserialize_vocabs(model_dict['vocab'], opt)
        class_dict = model_dict[cls.__name__]
        model = cls(vocabs=vocabs, opt=opt)

        pretrained_dict = class_dict['state_dict']

        # Only load dict that matches our model
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        
        # Load directly
        #model.load_state_dict(pretrained_dict)
        return model