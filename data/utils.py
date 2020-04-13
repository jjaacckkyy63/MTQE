import copy
import logging
from collections import defaultdict
from math import ceil
from pathlib import Path

import torch

from data.fieldsets.fieldset import Fieldset

# serialize or deserialize vocabs
def serialize_vocabs(vocabs, include_vectors=False):
    """Make vocab dictionary serializable.
       ('source': dict(word: idx), 'target': dict(word: idx))
    """
    serialized_vocabs = []

    for name, vocab in vocabs.items():
        vocab = copy.copy(vocab)
        vocab.stoi = dict(vocab.stoi)
        if not include_vectors:
            vocab.vectors = None
        serialized_vocabs.append((name, vocab))

    return serialized_vocabs

def deserialize_vocabs(vocabs):
    """Restore defaultdict lost in serialization.
    """
    vocabs = dict(vocabs)
    for name, vocab in vocabs.items():
        # Hack. Can't pickle defaultdict :(
        vocab.stoi = defaultdict(lambda: const.UNK_ID, vocab.stoi)
    return vocabs

# load vocab
def load_vocabularies_to_datasets(vocab_path, *datasets):
    fields = {}
    for dataset in datasets:
        fields.update(dataset.fields)
    return load_vocabularies_to_fields(vocab_path, fields)

def load_vocabularies_to_fields(vocab_path, fields):
    """Load serialized Vocabularies from disk into fields."""
    if Path(vocab_path).exists():
        vocabs_dict = torch.load(
            str(vocab_path), map_location=lambda storage, loc: storage
        )
        vocabs = vocabs_dict[const.VOCAB]
        fields = deserialize_fields_from_vocabs(fields, vocabs)
        #logger.info('Loaded vocabularies from {}'.format(vocab_path))
        return all(
            [vocab_loaded_if_needed(field) for _, field in fields.items()]
        )
    return False

def deserialize_fields_from_vocabs(fields, vocabs):
    """
    Load serialized vocabularies into their fields.
    """
    vocabs = deserialize_vocabs(vocabs)
    return fields_from_vocabs(fields, vocabs)


# load fields from vocab or opposite
def fields_from_vocabs(fields, vocabs):
    """
    Load Field objects from vocabs dict.
    From OpenNMT
    """
    vocabs = deserialize_vocabs(vocabs)
    for name, vocab in vocabs.items():
        if name not in fields:
            logger.debug(
                'No field "{}" for loading vocabulary; ignoring.'.format(name)
            )
        else:
            fields[name].vocab = vocab
    return fields

def fields_to_vocabs(fields):
    """
    Extract Vocab Dictionary from Fields Dictionary.
       Args:
          fields: A dict mapping field names to Field objects
       Returns:
          vocab: A dict mapping field names to Vocabularies
    """
    vocabs = {}
    for name, field in fields.items():
        if field is not None and 'vocab' in field.__dict__:
            vocabs[name] = field.vocab
    return vocabs

def filter_len(
    x,
    source_min_length=1,
    source_max_length=float('inf'),
    target_min_length=1,
    target_max_length=float('inf'),
):
    return (source_min_length <= len(x.source) <= source_max_length) and (
        target_min_length <= len(x.target) <= target_max_length
    )