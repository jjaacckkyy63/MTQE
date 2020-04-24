import torch
from torch import nn
from collections import OrderedDict

from models import Model
from models.utils import apply_packed_sequence, replace_token
from data.utils import deserialize_vocabs

class Attention(nn.Module):
    """Generic Attention Implementation.
       Module computes a convex combination of a set of values based on the fit
       of their keys with a query.
    """

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer
        self.mask = None

    def forward(self, query, keys, values=None):
        if values is None:
            values = keys
        scores = self.scorer(query, keys)
        # Masked Softmax
        scores = scores - scores.mean(1, keepdim=True)  # numerical stability
        scores = torch.exp(scores)
        if self.mask is not None:
            scores = self.mask * scores
        convex = scores / scores.sum(1, keepdim=True)
        return torch.einsum('bs,bsi->bi', [convex, values])

    def set_mask(self, mask):
        self.mask = mask

class Scorer(nn.Module):
    """Score function for Attention module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, keys):
        """Computes Scores for each key given the query.
           args:
                 query:  FloatTensor batch x n
                 keys:   FloatTensor batch x seq_length x m
           ret:
                 scores: FloatTensor batch x seq_length
        """
        raise NotImplementedError


class MLPScorer(Scorer):
    """Implements a score function based on a Multilayer Perceptron.
    """

    def __init__(self, query_size, key_size, layers=2, nonlinearity=nn.Tanh):
        super().__init__()
        layer_list = []
        size = query_size + key_size
        for i in range(layers):
            size_next = size // 2 if i < layers - 1 else 1
            layer_list.append(
                nn.Sequential(nn.Linear(size, size_next), nonlinearity())
            )
            size = size_next
        self.layers = nn.ModuleList(layer_list)

    def forward(self, query, keys):
        layer_in = torch.cat([query.unsqueeze(1).expand_as(keys), keys], dim=-1)
        layer_in = layer_in.reshape(-1, layer_in.size(-1))
        for layer in self.layers:
            layer_in = layer(layer_in)
        out = layer_in.reshape(keys.size()[:-1])
        return out


@Model.register_subclass
class BilstmPredictor(Model):
    """Bidirectional Conditional Language Model
       Implemented after Kim et al 2017, see:
         http://www.statmt.org/wmt17/pdf/WMT63.pdf
    """

    title = 'PredEst Predictor model (an embedder model)'

    def __init__(self, vocabs, opt, predict_inverse=False):
        """
        Args:
          vocabs: Dictionary Mapping Field Names to Vocabularies.
        kwargs:
          config: A state dict of a PredictorConfig object.
          dropout: LSTM dropout Default 0.0
          hidden_pred: LSTM Hidden Size, default 200
          rnn_layers: Default 3
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

        scorer = MLPScorer(
            opt.hidden_pred * 2, opt.hidden_pred * 2, layers=2
        )

        self.source_vocab_size = len(vocabs['source'])
        self.target_vocab_size = len(vocabs['target'])

        self.attention = Attention(scorer)
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
        self.lstm_source = nn.LSTM(
            input_size=opt.source_embeddings_size,
            hidden_size=opt.hidden_pred,
            num_layers=opt.rnn_layers_pred,
            batch_first=True,
            dropout=opt.dropout_pred,
            bidirectional=True,
        )
        self.forward_target = nn.LSTM(
            input_size=opt.target_embeddings_size,
            hidden_size=opt.hidden_pred,
            num_layers=opt.rnn_layers_pred,
            batch_first=True,
            dropout=opt.dropout_pred,
            bidirectional=False,
        )
        self.backward_target = nn.LSTM(
            input_size=opt.target_embeddings_size,
            hidden_size=opt.hidden_pred,
            num_layers=opt.rnn_layers_pred,
            batch_first=True,
            dropout=opt.dropout_pred,
            bidirectional=False,
        )

        self.W1 = self.embedding_target
        if not opt.share_embeddings:
            self.W1 = nn.Embedding(
                self.target_vocab_size,
                opt.out_embeddings_size,
                opt.PAD_ID,
            )
        self.W2 = nn.Parameter(
            torch.zeros(
                opt.out_embeddings_size, opt.out_embeddings_size
            )
        )
        self.V = nn.Parameter(
            torch.zeros(
                2 * opt.target_embeddings_size,
                2 * opt.out_embeddings_size,
            )
        )
        self.C = nn.Parameter(
            torch.zeros(
                2 * opt.hidden_pred, 2 * opt.out_embeddings_size
            )
        )
        self.S = nn.Parameter(
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
    def from_options(cls, vocabs, opt, PreModelClass=None):
        return cls(vocabs, opt)
    
    # Load other model path
    @classmethod
    def from_file(cls, path, opt):
        model_dict = torch.load(
            str(path), map_location=lambda storage, loc: storage
        )
        if cls.__name__ not in model_dict:
            raise KeyError(
                '{} model data not found in {}'.format(cls.__name__, path)
            )

        return cls.from_dict(model_dict, opt)
    
    @classmethod
    def from_dict(cls, model_dict, opt, PreModelClass=None):
        vocabs = deserialize_vocabs(model_dict['vocab'], opt)
        class_dict = model_dict[cls.__name__]
        model = cls(vocabs=vocabs, opt=opt)
        model.load_state_dict(class_dict['state_dict'])
        return model

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

    def forward(self, batch, source_side=None, target_side=None):
        if not source_side:
            source_side = self.source_side
        if not target_side:
            target_side = self.target_side

        source = getattr(batch, source_side)
        target = getattr(batch, target_side)

        batch_size, target_len = target.shape[:2]
        # Remove First and Last Element (Start / Stop Tokens)
        source_mask = self.get_mask(batch, source_side)[:, 1:-1]
        source_lengths = source_mask.sum(1)
        target_lengths = self.get_mask(batch, target_side).sum(1)
        source_embeddings = self.embedding_source(source)
        target_embeddings = self.embedding_target(target)
        # Source Encoding
        source_contexts, hidden = apply_packed_sequence(
            self.lstm_source, source_embeddings, source_lengths
        )
        # Target Encoding.
        h_forward, h_backward = self._split_hidden(hidden)
        forward_contexts, _ = self.forward_target(target_embeddings, h_forward)
        target_emb_rev = self._reverse_padded_seq(
            target_lengths, target_embeddings
        )
        backward_contexts, _ = self.backward_target(target_emb_rev, h_backward)
        backward_contexts = self._reverse_padded_seq(
            target_lengths, backward_contexts
        )

        # For each position, concatenate left context i-1 and right context i+1
        target_contexts = torch.cat(
            [forward_contexts[:, :-2], backward_contexts[:, 2:]], dim=-1
        )
        # For each position i, concatenate Emeddings i-1 and i+1
        target_embeddings = torch.cat(
            [target_embeddings[:, :-2], target_embeddings[:, 2:]], dim=-1
        )

        # Get Attention vectors for all positions and stack.
        self.attention.set_mask(source_mask.float())
        attns = [
            self.attention(
                target_contexts[:, i], source_contexts, source_contexts
            )
            for i in range(target_len - 2)
        ]
        attns = torch.stack(attns, dim=1)

        # Combine attention, embeddings and target context vectors
        C = torch.einsum('bsi,il->bsl', [attns, self.C])
        V = torch.einsum('bsj,jl->bsl', [target_embeddings, self.V])
        S = torch.einsum('bsk,kl->bsl', [target_contexts, self.S])
        t_tilde = C + V + S
        # Maxout with pooling size 2
        t, _ = torch.max(
            t_tilde.view(
                t_tilde.shape[0], t_tilde.shape[1], t_tilde.shape[-1] // 2, 2
            ),
            dim=-1,
        )

        f = torch.einsum('oh,bso->bsh', [self.W2, t])
        logits = torch.einsum('vh,bsh->bsv', [self.W1.weight, f])
        PreQEFV = torch.einsum('bsh,bsh->bsh', [self.W1(target[:, 1:-1]), f])
        PostQEFV = torch.cat([forward_contexts, backward_contexts], dim=-1)
        return {
            target_side: logits,
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
    def _split_hidden(hidden):
        """Split Hidden State into forward/backward parts.
        """
        h, c = hidden
        size = h.shape[0]
        idx_forward = torch.arange(0, size, 2, dtype=torch.long)
        idx_backward = torch.arange(1, size, 2, dtype=torch.long)
        hidden_forward = (h[idx_forward], c[idx_forward])
        hidden_backward = (h[idx_backward], c[idx_backward])
        return hidden_forward, hidden_backward

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
