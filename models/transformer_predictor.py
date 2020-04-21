import torch
from torch import nn
from collections import OrderedDict
import math

from models import Model, NCELoss
from models.utils import apply_packed_sequence, replace_token
from data.utils import deserialize_vocabs

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

@Model.register_subclass
class TransformerPredictor(Model):
    
    title = 'Transformer based model (an embedder model)'

    def __init__(self, vocabs, opt, idx2count=None, predict_inverse=False):
        """
        Args:
          vocabs: Dictionary Mapping Field Names to Vocabularies.
        kwargs:
          config: A state dict of a PredictorConfig object.
          dropout: Transformer dropout Default 0.1
          hidden_pred: LSTM Hidden Size, default 400
          number of heads: Default 8
          transformer_layers: Default 6
          embedding_sizes: If set, takes precedence over other embedding params
                           Default 100
          source_embeddings_size: Default 100
          target_embeddings_size: Default 100
          out_embeddings_size: Output softmax embedding. Default 100
          share_embeddings: Tie input and output embeddings for target.
                            Default False
          predict_inverse: Predict from target to source. Default False
        """
        super().__init__(vocabs=vocabs, opt=opt)

        self.source_vocab_size = len(vocabs['source'])
        self.target_vocab_size = len(vocabs['target'])
        self.hidden_pred = opt.hidden_pred

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.embedding_source = nn.Embedding(
            self.source_vocab_size,
            opt.source_embeddings_size,
            opt.PAD_ID,
        )
        self.embedding_target = nn.Embedding(
            self.target_vocab_size,
            opt.target_embeddings_size,
            opt.PAD_ID,
        )
        self.source_hidden = nn.Linear(
            opt.source_embeddings_size, 
            opt.hidden_pred
        )
        self.target_hidden = nn.Linear(
            opt.target_embeddings_size, 
            opt.hidden_pred
        )
        self.pos_encoder_source = PositionalEncoding(
            opt.hidden_pred, 
            opt.tdropout_pred
        )
        self.pos_encoder_target = PositionalEncoding(
            opt.hidden_pred, 
            opt.tdropout_pred
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=opt.hidden_pred,
            nhead=opt.nhead,
            dropout=opt.tdropout_pred
        )
        self.forward_decoder_layer = nn.TransformerDecoderLayer(
            d_model=opt.hidden_pred,
            nhead=opt.nhead,
            dropout=opt.tdropout_pred,
        )
        self.backward_decoder_layer = nn.TransformerDecoderLayer(
            d_model=opt.hidden_pred,
            nhead=opt.nhead,
            dropout=opt.tdropout_pred,
        )
        self.encoder_source = nn.TransformerEncoder(self.encoder_layer, num_layers=opt.transformer_layers_pred)
        self.forward_decoder_target = nn.TransformerDecoder(self.forward_decoder_layer, num_layers=opt.transformer_layers_pred)
        self.backward_decoder_target = nn.TransformerDecoder(self.backward_decoder_layer, num_layers=opt.transformer_layers_pred)

        self.W1 = self.embedding_target
        if not opt.share_embeddings:
            self.W1 = nn.Embedding(
                self.target_vocab_size,
                opt.out_embeddings_size,
                opt.PAD_ID,
            )
        self.W2 = nn.Parameter(
            torch.zeros(
                4 * opt.out_embeddings_size, opt.out_embeddings_size
            )
        )
        self.V = nn.Parameter(
            torch.zeros(
                2 * opt.hidden_pred,
                2 *opt.out_embeddings_size,
            )
        )
        self.C = nn.Parameter(
            torch.zeros(
                2 * opt.hidden_pred, 2 * opt.out_embeddings_size
            )
        )
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        self._loss = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=opt.PAD_ID
        )
        
        self._nceloss = NCELoss(
            idx2count,
            noise_ratio=30,
            loss_type='nce',
            reduction='elementwise_mean',
            target_vocab_size=self.target_vocab_size,
            opt=opt)

        self.opt = opt

        self.source_side, self.target_side = (
                self.opt.source_side,
                self.opt.target_side,
            )
        

        if predict_inverse:
            self.source_side, self.target_side = (
                self.opt.target_side,
                self.opt.source_side,
            )
            self.target_vocab_size, self.source_vocab_size = (
                self.source_vocab_size,
                self.target_vocab_size,
            )

    @classmethod
    def from_options(cls, vocabs, opt, PreModelClass=None, idx2count=None):
        return cls(vocabs, opt, idx2count)
    
    # Load other model path
    @classmethod
    def from_file(cls, path, opt, idx2count):
        model_dict = torch.load(
            str(path), map_location=lambda storage, loc: storage
        )
        if cls.__name__ not in model_dict:
            raise KeyError(
                '{} model data not found in {}'.format(cls.__name__, path)
            )

        return cls.from_dict(model_dict, opt, idx2count=idx2count)
    
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
    

    def loss(self, model_out, batch, target_side=None):
        if not target_side:
            target_side = self.target_side
        target = getattr(batch, target_side)
        # There are no predictions for first/last element
        target = replace_token(target[:, 1:-1], self.opt.STOP_ID, self.opt.PAD_ID)
        # Predicted Class must be in dim 1 for xentropyloss
        
        # ce loss
        logits = model_out[target_side]
        logits = logits.transpose(1, 2)
        loss = self._loss(logits, target)
        
        # nce loss
        # logits = model_out['target_side_hidden']
        # loss = self._nceloss(target, logits)
        
        loss_dict = OrderedDict()
        loss_dict[target_side] = loss
        loss_dict['loss'] = loss
        return loss_dict

    def forward(self, batch, source_side=None, target_side=None):
        if not source_side:
            source_side = self.source_side
        if not target_side:
            target_side = self.target_side

        source = getattr(batch, source_side)
        target = getattr(batch, target_side)
        batch_size, target_len = target.shape[:2]

        source_mask = (1-self.get_mask(batch, source_side)).bool()
        target_mask = (1-self.get_mask(batch, target_side)).bool()
        source_lengths = self.get_mask(batch, source_side)[:, 1:-1].sum(1)
        target_lengths = self.get_mask(batch, target_side).sum(1)
        # print(source[0])
        # print(target[0])
        # print(source_mask[0])
        # print(target_mask[0])
        # print(source_lengths[0])
        # print(target_lengths[0])
        
        source_embeddings = self.embedding_source(source)
        source_embeddings = self.source_hidden(source_embeddings)*math.sqrt(self.hidden_pred)
        source_embeddings = self.pos_encoder_source(source_embeddings)

        target_embeddings = self.embedding_target(target)
        target_embeddings = self.target_hidden(target_embeddings)*math.sqrt(self.hidden_pred)
        target_embeddings = self.pos_encoder_target(target_embeddings)

        ################## Transformer process ##################
        source_embeddings_t = source_embeddings.permute(1,0,2)
        target_embeddings_t = target_embeddings.permute(1,0,2)
        target_attention_mask = self.src_mask(target_embeddings_t.shape[0]).to(self.device)

        # Source Encoding
        hidden = self.encoder_source(src = source_embeddings_t, 
                                     src_key_padding_mask = source_mask)
        # Target Encoding.
        forward_contexts = self.forward_decoder_target(tgt = target_embeddings_t,
                                                       memory = hidden,
                                                       tgt_key_padding_mask = target_mask,
                                                       memory_key_padding_mask = source_mask
                                                       )
        
        target_emb_rev = self._reverse_padded_seq(target_lengths, target_embeddings).permute(1,0,2)
        target_mask_rev = self._reverse_padded_seq(target_lengths, target_mask.unsqueeze(-1)).squeeze(-1)
        backward_contexts = self.backward_decoder_target(tgt = target_emb_rev, 
                                                         memory = hidden,
                                                         tgt_key_padding_mask = target_mask_rev,
                                                         memory_key_padding_mask = source_mask
                                                         )
        
        ################## Transformer process ##################

        source_contexts = hidden.permute(1,0,2)
        forward_contexts = forward_contexts.permute(1,0,2)
        backward_contexts = backward_contexts.permute(1,0,2)
        backward_contexts = self._reverse_padded_seq(target_lengths, backward_contexts)

        # For each position, concatenate left context i-1 and right context i+1
        target_contexts = torch.cat(
            [forward_contexts[:, :-2], backward_contexts[:, 2:]], dim=-1
        )
        # For each position i, concatenate Emeddings i-1 and i+1
        target_embeddings = torch.cat(
            [target_embeddings[:, :-2], target_embeddings[:, 2:]], dim=-1
        )
        
        # Combine attention, embeddings and target context vectors
        C = torch.einsum('bsi,il->bsl', [target_contexts, self.C])
        V = torch.einsum('bsj,jl->bsl', [target_embeddings, self.V])
        _pre_qefv = torch.cat([C, V], dim=-1)

        f = torch.einsum('oh,bso->bsh', [self.W2, _pre_qefv])
        logits = torch.einsum('vh,bsh->bsv', [self.W1.weight, f])

        PreQEFV = torch.einsum('bsh,bsh->bsh', [self.W1(target[:, 1:-1]), f])
        PostQEFV = torch.cat([forward_contexts, backward_contexts], dim=-1)
        return {
            target_side: logits,
            'target_side_hidden': f,
            'PREQEFV': PreQEFV,
            'POSTQEFV': PostQEFV,
        }

    @staticmethod
    def _reverse_padded_seq(lengths, sequence):
        """ Reverses a batch of padded sequences of different length.
        """
        batch_size, max_length = sequence.shape[:-1]
        reversed_idx = []
        for i in range(batch_size * max_length):
            batch_id = i // max_length
            sent_id = i % max_length
            if sent_id < lengths[batch_id]:
                sent_id_rev = lengths[batch_id] - sent_id - 1
            else:
                sent_id_rev = sent_id  # Padding symbol, don't change order
            reversed_idx.append(max_length * batch_id + sent_id_rev)
        flat_sequence = sequence.contiguous().view(batch_size * max_length, -1)
        reversed_seq = flat_sequence[reversed_idx, :].view(*sequence.shape)
        return reversed_seq
    
    @staticmethod
    def src_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def metrics(self):
        metrics = []

        main_metric = PerplexityMetric(
            prefix=opt.target_side,
            target_name=opt.target_side,
            PAD=opt.PAD_ID,
            STOP=opt.STOP_ID,
        )
        metrics.append(main_metric)

        metrics.append(
            CorrectMetric(
                prefix=opt.target_side,
                target_name=opt.target_side,
                PAD=opt.PAD_ID,
                STOP=opt.STOP_ID,
            )
        )
        metrics.append(
            ExpectedErrorMetric(
                prefix=opt.target_side,
                target_name=opt.target_side,
                PAD=opt.PAD_ID,
                STOP=opt.STOP_ID,
            )
        )
        return metrics

    def metrics_ordering(self):
        return min