import torch
from torch import nn
from collections import OrderedDict
from torch.distributions.normal import Normal

from models import Model, BilstmPredictor, TransformerPredictor
from models.utils import apply_packed_sequence, make_loss_weights
from data.utils import deserialize_vocabs

@Model.register_subclass
class Estimator(Model):
    title = 'PredEst Estimator model'

    def __init__(
        self, vocabs, opt, predictor_tgt=None, predictor_src=None, PreModelClass='TransformerPredictor'
        ):

        super().__init__(vocabs=vocabs, opt=opt)

        if not predictor_tgt:
            if opt.load_pred_target:
                predictor_tgt = eval(PreModelClass).from_file(opt.load_pred_target, opt)
            else:
                predictor_tgt = eval(PreModelClass)(vocabs, opt, predict_inverse=False)
        
        if not predictor_src:
            if opt.load_pred_source:
                predictor_src = eval(PreModelClass).from_file(opt.load_pred_source, opt)
            else:
                predictor_src = eval(PreModelClass)(vocabs, opt, predict_inverse=True)
        
        if opt.token_level:
            if predictor_src:
                predictor_src.vocabs = vocabs
            if predictor_tgt:
                predictor_tgt.vocabs = vocabs
        
        self.predictor_tgt = predictor_tgt
        self.predictor_src = predictor_src

        self.mlp = None
        self.sentence_pred = None
        self.sentence_sigma = None
        self.lstm_input_size = 2 * opt.hidden_pred + opt.out_embeddings_size

        if opt.mlp_est:
            self.mlp = nn.Sequential(
                nn.Linear(self.lstm_input_size, opt.hidden_est), nn.Tanh()
            )
            self.lstm_input_size = opt.hidden_est

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.opt.hidden_est,
            num_layers=self.opt.rnn_layers_est,
            batch_first=True,
            dropout=self.opt.dropout_est,
            bidirectional=True,
        )

        sentence_input_size = 2 * opt.rnn_layers_est * opt.hidden_est
        self.sentence_pred = nn.Sequential(
                nn.Linear(sentence_input_size, sentence_input_size // 2),
                nn.Sigmoid(),
                nn.Linear(sentence_input_size // 2, sentence_input_size // 4),
                nn.Sigmoid(),
                nn.Linear(sentence_input_size // 4, 1),
            )
        if self.opt.sentence_ll:
            # Predict truncated Gaussian distribution
            self.sentence_sigma = nn.Sequential(
                nn.Linear(sentence_input_size, sentence_input_size // 2),
                nn.Sigmoid(),
                nn.Linear(
                    sentence_input_size // 2, sentence_input_size // 4
                ),
                nn.Sigmoid(),
                nn.Linear(sentence_input_size // 4, 1),
                nn.Sigmoid(),
            )

        self.mse_loss = nn.MSELoss(reduction='sum')

        if opt.start_stop:
            self.start_PreQEFV = nn.Parameter(torch.zeros(1, 1, opt.out_embeddings_size))
            self.end_PreQEFV = nn.Parameter(torch.zeros(1, 1, opt.out_embeddings_size))

        self.opt = opt
    
    @classmethod
    def from_options(cls, vocabs, opt, PreModelClass='TransformerPredictor'):
        predictor_src = predictor_tgt = None
        if opt.load_pred_source:
            predictor_src = eval(PreModelClass).from_file(opt.load_pred_source, opt)
        if opt.load_pred_target:
            predictor_tgt = eval(PreModelClass).from_file(opt.load_pred_target, opt)

        return cls(vocabs, opt, 
                predictor_tgt=predictor_tgt, predictor_src=predictor_src, PreModelClass=PreModelClass)
    
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
    def from_dict(cls, model_dict, opt, PreModelClass=None, vocabs=None):
        vocabs = deserialize_vocabs(model_dict['vocab'], opt)
        class_dict = model_dict[cls.__name__]
        model = cls(vocabs=vocabs, opt=opt, PreModelClass=PreModelClass)
        model.load_state_dict(class_dict['state_dict'])
        return model
    
    def loss(self, model_out, batch):

        sentence_pred = model_out['scores']
        sentence_scores = batch.sentences_scores

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
    
    def make_input(self, model_out, batch, side='target'):

        PreQEFV = model_out['PREQEFV']
        PostQEFV = model_out['POSTQEFV']

        token_mask = self.get_mask(batch, side)
        batch_size = token_mask.shape[0]
        target_lengths = token_mask.sum(1)

        if self.opt.start_stop:
            target_lengths += 2
            start = self.start_PreQEFV.expand(
                batch_size, 1, self.opt.out_embeddings_size
            )
            end = self.end_PreQEFV.expand(
                batch_size, 1, self.opt.out_embeddings_size
            )
            PreQEFV = torch.cat((start, PreQEFV, end), dim=1)
        else:
            PostQEFV = PostQEFV[:, 1:-1]
        
        input_seq = torch.cat([PreQEFV, PostQEFV], dim=-1)
        length, input_dim = input_seq.shape[1:]
        if self.mlp:
            input_flat = input_seq.view(batch_size * length, input_dim)
            input_flat = self.mlp(input_flat)
            input_seq = input_flat.view(
                batch_size, length, self.lstm_input_size
            )
        return input_seq, target_lengths
    
    def make_sentence_input(self, h_tgt, h_src):
        """Reshape last hidden state. """
        h = h_tgt[0] if h_tgt else h_src[0]
        h = h.contiguous().transpose(0, 1)
        return h.reshape(h.shape[0], -1)
    
    def predict_sentence(self, sentence_input):
        """Compute Sentence Score predictions."""
        outputs = OrderedDict()
        sentence_scores = self.sentence_pred(sentence_input).squeeze()
        outputs['scores'] = sentence_scores
        if self.sentence_sigma:
            # Predict truncated Gaussian on [0,1]
            sigma = self.sentence_sigma(sentence_input).squeeze()
            
            outputs['sent_mu'] = outputs['scores']
            outputs['sent_sigma'] = sigma
            mean = outputs['sent_mu'].clone().detach()
            
            # Compute log-likelihood of x given mu, sigma
            normal = Normal(mean, sigma)
            # Renormalize on [0,1] for truncated Gaussian
            partition_function = (normal.cdf(1) - normal.cdf(0)).detach()
            outputs['scores'] = mean + (
                (
                    sigma ** 2
                    * (normal.log_prob(0).exp() - normal.log_prob(1).exp())
                )
                / partition_function
            )

        return outputs
    
    def forward(self, batch):

        contexts_tgt, h_tgt = None, None
        contexts_src, h_src = None, None

        # Predict Target from Source
        model_out_tgt = self.predictor_tgt(batch)
        input_seq, target_lengths = self.make_input(
            model_out_tgt, batch, 'target'
        )
        contexts_tgt, h_tgt = apply_packed_sequence(
            self.lstm, input_seq, target_lengths
        )

        # Predict Source from Target
        #model_out_src = self.predictor_src(batch)
        #input_seq, target_lengths = self.make_input(
        #    model_out_src, batch, 'source'
        #)
        #contexts_src, h_src = apply_packed_sequence(
        #    self.lstm, input_seq, target_lengths
        #)

        sentence_input = self.make_sentence_input(h_tgt, h_src)
        outputs = self.predict_sentence(sentence_input)

        return outputs























