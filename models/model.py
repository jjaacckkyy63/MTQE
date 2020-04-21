import torch
import torch.nn as nn
import math

from data.utils import *

class Model(nn.Module):

    subclasses = {}

    def __init__(self, vocabs, opt, idx2count=None):
    
        super(Model, self).__init__()

        self.vocabs = vocabs
        self.opt = opt

    @classmethod
    def register_subclass(cls, subclass):
        cls.subclasses[subclass.__name__] = subclass
        return subclass

    def loss(self, model_out, target):
        pass

    def forward(self, *args, **kwargs):
        pass

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_mask(self, batch, output):
        """Compute Mask of Tokens for side.
        Args:
            batch: Namespace of tensors
            side: String identifier.
        """
        side = output

        input_tensor = getattr(batch, side)
        if isinstance(input_tensor, tuple) and len(input_tensor) == 2:
            input_tensor, lengths = input_tensor

        mask = torch.ones_like(input_tensor, dtype=torch.uint8)

        possible_padding = [self.opt.PAD, self.opt.START, self.opt.STOP]
        unk_id = self.vocabs[side].stoi.get(self.opt.UNK)
        for pad in possible_padding:
            pad_id = self.vocabs[side].stoi.get(pad)
            if pad_id is not None and pad_id != unk_id:
                mask &= torch.as_tensor(
                    input_tensor != pad_id,
                    device=mask.device,
                    dtype=torch.uint8,
                )

        return mask
    
    # Load main model path
    @staticmethod
    def create_from_file(path, opt, vocabs=None, idx2count=None):
    
        try:
            model_dict = torch.load(path, map_location=lambda s,l: s)
        except FileNotFoundError:
            # If no model is found
            raise FileNotFoundError(
                'No valid model data found in {}'.format(path)
            )

        for model_name in Model.subclasses:
            if 'Predictor' in model_dict:
                model_dict[model_name] = model_dict['Predictor']
            if model_name in model_dict:
                model = Model.subclasses[model_name].from_dict(model_dict, opt, PreModelClass=opt.pre_model_name, vocabs=vocabs, idx2count=idx2count)
                return model
    
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
    def from_dict(cls, model_dict, opt, idx2count=None):
        vocabs = deserialize_vocabs(model_dict['vocab'], opt)
        class_dict = model_dict[cls.__name__]
        model = cls(vocabs=vocabs, opt=opt, idx2count=idx2count)
        model.load_state_dict(class_dict['state_dict'])
        return model
    
    def save(self, path):
        vocabs = serialize_vocabs(self.vocabs)
        model_dict = {
            'vocab': vocabs,
            self.__class__.__name__: {
                'state_dict': self.state_dict(),
            },
        }
        torch.save(model_dict, path)


########################## NCE Loss ##########################

class NCELoss(nn.Module):
    """Noise Contrastive Estimation
    There are 3 loss modes in this NCELoss module:
        - nce: enable the NCE approximation
        - sampled: enabled sampled softmax approximation
        - full: use the original cross entropy as default loss
    They can be switched by directly setting `nce.loss_type = 'nce'`.
    
    Attributes:
        noise: the distribution of noise
        noise_ratio: $\frac{#noises}{#real data samples}$ (k in paper)
        norm_term: the normalization term (lnZ in paper), can be heuristically
        determined by the number of classes, plz refer to the code.
        reduction: reduce methods, same with pytorch's loss framework, 'none',
        'elementwise_mean' and 'sum' are supported.
        loss_type: loss type of this module, currently 'full', 'sampled', 'nce'
        are supported
    Shape:
        - noise: :math:`(V)` where `V = vocabulary size`
        - target: :math:`(B, N)`
        - loss: a scalar loss by default, :math:`(B, N)` if `reduction='none'`
    Input:
        target: the supervised training label.
        args&kwargs: extra arguments passed to underlying index module
    Return:
        loss: if `reduction='sum' or 'elementwise_mean'` the scalar NCELoss ready for backward,
        else the loss matrix for every individual targets.
    """
    
    def __init__(self,
                 idx2count,
                 noise_ratio=100,
                 norm_term='auto',
                 reduction='elementwise_mean',
                 per_word=False,
                 loss_type='nce',
                 target_vocab_size=None,
                 opt=None
                 ):
        super(NCELoss, self).__init__()

        # Re-norm the given noise frequency list and compensate words with
        # extremely low prob for numeric stability
        noise = self.build_unigram_noise(
        torch.FloatTensor(idx2count)
        )
        probs = noise / noise.sum()
        probs = probs.clamp(min=1e-10)
        renormed_probs = probs / probs.sum()

        self.register_buffer('logprob_noise', renormed_probs.log())
        self.alias = AliasMultinomial(renormed_probs)

        self.noise_ratio = noise_ratio
        if norm_term == 'auto':
            self.norm_term = math.log(noise.numel())
        else:
            self.norm_term = norm_term
        self.reduction = reduction
        self.per_word = per_word
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type
        
        self.emb = nn.Embedding(target_vocab_size, opt.out_embeddings_size) #opt.hidden_pred
        self.bias = nn.Embedding(target_vocab_size, 1)
    
    def forward(self, target, *args, **kwargs):
        """compute the loss with output and the desired target
        The `forward` is the same among all NCELoss submodules, it
        takes care of generating noises and calculating the loss
        given target and noise scores.
        """

        batch = target.size(0)
        max_len = target.size(1)
        if self.loss_type != 'full':

            noise_samples = self.get_noise(batch, max_len)

            # B,N,Nr
            logit_noise_in_noise = self.logprob_noise[noise_samples.data.view(-1)].view_as(noise_samples)
            logit_target_in_noise = self.logprob_noise[target.data.view(-1)].view_as(target)

            # (B,N), (B,N,Nr)
            logit_target_in_model, logit_noise_in_model = self._get_logit(target, noise_samples, *args, **kwargs)

            if self.loss_type == 'nce':
                if self.training:
                    loss = self.nce_loss(
                        logit_target_in_model, logit_noise_in_model,
                        logit_noise_in_noise, logit_target_in_noise,
                    )
                else:
                    # directly output the approximated posterior
                    loss = - logit_target_in_model
            elif self.loss_type == 'sampled':
                loss = self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
            # NOTE: The mix mode is still under investigation
            elif self.loss_type == 'mix' and self.training:
                loss = 0.5 * self.nce_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )
                loss += 0.5 * self.sampled_softmax_loss(
                    logit_target_in_model, logit_noise_in_model,
                    logit_noise_in_noise, logit_target_in_noise,
                )

            else:
                current_stage = 'training' if self.training else 'inference'
                raise NotImplementedError(
                    'loss type {} not implemented at {}'.format(
                        self.loss_type, current_stage
                    )
                )

        else:
            # Fallback into conventional cross entropy
            loss = self.ce_loss(target, *args, **kwargs)

        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_noise(self, batch_size, max_len):
        """Generate noise samples from noise distribution"""

        noise_size = (batch_size, max_len, self.noise_ratio)
        if self.per_word:
            noise_samples = self.alias.draw(*noise_size)
        else:
            noise_samples = self.alias.draw(1, 1, self.noise_ratio).expand(*noise_size)

        noise_samples = noise_samples.contiguous()
        return noise_samples

    def _get_logit(self, target_idx, noise_idx, *args, **kwargs):
        """Get the logits of NCE estimated probability for target and noise
        Both NCE and sampled softmax Loss are unchanged when the probabilities are scaled
        evenly, here we subtract the maximum value as in softmax, for numeric stability.
        Shape:
            - Target_idx: :math:`(N)`
            - Noise_idx: :math:`(N, N_r)` where `N_r = noise ratio`
        """

        target_logit, noise_logit = self.get_score(target_idx, noise_idx, *args, **kwargs)

        target_logit = target_logit.sub(self.norm_term)
        noise_logit = noise_logit.sub(self.norm_term)
        return target_logit, noise_logit
    
    def get_score(self, target_idx, noise_idx, input):
        """
        Shape:
            - target_idx: :math:`(B, L)` where `B` is batch size
            `L` is sequence length
            - noise_idx: :math:`(B, L, N_r)` where `N_r is noise ratio`
            - input: :math:`(B, L, E)` where `E = output embedding size`
        """

        if self.per_word:
            return self._compute_sampled_logit(
                target_idx, noise_idx, input
            )
        else:
            return self._compute_sampled_logit_batched(
                target_idx, noise_idx, input
            )

    def _compute_sampled_logit(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector
        Args:
            - target_idx: :math:`B, L, 1`
            - noise_idx: :math:`B, L, N_r` target_idx and noise_idx are
            concatenated into one single index matrix for performance
            - input: :math:`(B, L, E)` where `E = vector dimension`
        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        # the size will be used to pack the output of indexlinear
        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, 1, input.size(-1))  # N,1,E
        target_idx = target_idx.view(-1).unsqueeze(-1)  # N,1
        noise_idx = noise_idx.view(-1, noise_idx.size(-1))  # N,Nr
        indices = torch.cat([target_idx, noise_idx], dim=-1)

        target_batch = self.emb.weight.index_select(0, indices.view(-1)).view(*indices.size(), -1)
        bias = self.bias.weight.index_select(0, indices.view(-1)).view_as(indices)
        # the element-wise multiplication is automatically broadcasted
        logits = torch.sum(input * target_batch, dim=2) + bias
        logits = logits.view(*original_size, -1)

        target_score, noise_score = logits[:, :, 0], logits[:, :, 1:]
        return target_score, noise_score

    def _compute_sampled_logit_batched(self, target_idx, noise_idx, input):
        """compute the logits of given indices based on input vector
        A batched version, it speeds up computation and puts less burden on
        sampling methods.
        Args:
            - target_idx: :math:`B, L, 1` flatten to `(N)` where `N=BXL`
            - noise_idx: :math:`B, L, N_r`, noises at the dim along B and L
            should be the same, flatten to `N_r`
            - input: :math:`(B, L, E)` where `E = vector dimension`
        Returns:
            - target_score: :math:`(B, L)` the computed logits of target_idx
            - noise_score: :math:`(B, L, N_r)` the computed logits of noise_idx
        """

        original_size = target_idx.size()

        # flatten the following matrix
        input = input.contiguous().view(-1, input.size(-1))
        target_idx = target_idx.view(-1)
        noise_idx = noise_idx[0, 0].view(-1)

        target_batch = self.emb(target_idx)
        
        # target_bias = self.bias.index_select(0, target_idx)  # N
        target_bias = self.bias(target_idx).squeeze(1)  # N
        target_score = torch.sum(input * target_batch, dim=1) + target_bias  # N X E * N X E

        noise_batch = self.emb(noise_idx)  # Nr X H
        # noise_bias = self.bias.index_select(0, noise_idx).unsqueeze(0)  # Nr
        noise_bias = self.bias(noise_idx)  # 1, Nr
        noise_score = torch.matmul(
            input, noise_batch.t()
        ) + noise_bias.t()  # N X Nr
        return target_score.view(original_size), noise_score.view(*original_size, -1)
    
    def nce_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the classification loss given all four probabilities
        Args:
            - logit_target_in_model: logit of target words given by the model (RNN)
            - logit_noise_in_model: logit of noise words given by the model
            - logit_noise_in_noise: logit of noise words given by the noise distribution
            - logit_target_in_noise: logit of target words given by the noise distribution
        Returns:
            - loss: a mis-classification loss for every single case
        """

        logit_model = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        logit_noise = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)

        logit_true = logit_model - logit_noise - math.log(self.noise_ratio)

        label = torch.zeros_like(logit_model)
        label[:, :, 0] = 1

        loss = self.bce_with_logits(logit_true, label).sum(dim=2)
        return loss

    def sampled_softmax_loss(self, logit_target_in_model, logit_noise_in_model, logit_noise_in_noise, logit_target_in_noise):
        """Compute the sampled softmax loss based on the tensorflow's impl"""
        logits = torch.cat([logit_target_in_model.unsqueeze(2), logit_noise_in_model], dim=2)
        q_logits = torch.cat([logit_target_in_noise.unsqueeze(2), logit_noise_in_noise], dim=2)
        # subtract Q for correction of biased sampling
        logits = logits - q_logits
        labels = torch.zeros_like(logits.narrow(2, 0, 1)).squeeze(2).long()
        loss = self.ce(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        ).view_as(labels)

        return loss
    
    @staticmethod
    def build_unigram_noise(freq):
        """build the unigram noise from a list of frequency
        Parameters:
            freq: a tensor of #occurrences of the corresponding index
        Return:
            unigram_noise: a torch.Tensor with size ntokens,
            elements indicate the probability distribution
        """
        total = freq.sum()
        noise = freq / total
        assert abs(noise.sum() - 1) < 0.001
        return noise

    

    





    


