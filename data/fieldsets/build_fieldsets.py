import torch
from torchtext import data
import spacy
import re
import jieba

from data.fieldsets.fieldset import Fieldset

def tokenizer(text):
    """Implement your own tokenize procedure."""
    # nlp = spacy.load("xx_ent_wiki_sm")
    # return [tok.text for tok in nlp.tokenizer(text)]
    # return text.strip().split()
    
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;.#+_\"\']')
    text = re.sub(REPLACE_BY_SPACE_RE, '', text)

    if re.search('[\u4e00-\u9fa5]', text) is not None:
        tokens = [word for word in jieba.cut(text) if word.strip()]
    else:
        text = text.lower()
        tokens = text.strip().split()
    
    return tokens

def build_text_field(opt):
    return data.Field(
        tokenize=tokenizer,
        init_token=opt.START,
        batch_first=True,
        eos_token=opt.STOP,
        pad_token=opt.PAD,
        unk_token=opt.UNK,
    )

def build_fieldset(opt):
    
    target_field = build_text_field(opt)
    source_field = build_text_field(opt)

    source_vocab_options = dict(
        min_freq='source_vocab_min_frequency', max_size='source_vocab_size'
    )
    target_vocab_options = dict(
        min_freq='target_vocab_min_frequency', max_size='target_vocab_size'
    )
    source_vector_options = 'source-embeddings'
    target_vector_options = 'target-embeddings'

    fieldset = Fieldset()
    fieldset.add(
        name='source',
        field=source_field,
        vocab_options=source_vocab_options,
        vocab_vectors=source_vector_options
    )
    fieldset.add(
        name='target',
        field=target_field,
        vocab_options=target_vocab_options,
        vocab_vectors=target_vector_options
    )
    fieldset.add(
        name='sentences_scores',
        field=data.Field(
            sequential=False, use_vocab=False, dtype=torch.float32
        )
    )

    return fieldset