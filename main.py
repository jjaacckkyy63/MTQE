import torch
import numpy as np

from data.fieldsets.build_fieldsets import build_fieldset
from data.builders import build_training_datasets
from data.iterators import build_bucket_iterator
from data.utils import *
from trainers.utils import retrieve_trainer
from config import opt
from models import Model, BilstmPredictor, Estimator

def train():

    device = torch.cuda() if torch.cuda.is_available() else torch.device('cpu')

    # Data
    fieldset = build_fieldset(opt)
    
    train_dataset, valid_dataset = build_training_datasets(fieldset, opt, has_valid=True)

    vocabs = fields_to_vocabs(train_dataset.fields)
    # Call vocabulary
    #print(fieldset.fields['target'].vocab.itos)
    
    # Trainer
    ModelClass = eval(opt.model_name)
    trainer = retrieve_trainer(ModelClass, opt, vocabs, device)

    # Dataset iterators
    train_iter = build_bucket_iterator(
        train_dataset,
        batch_size=opt.train_batch_size,
        is_train=True,
        device=device
    )
    valid_iter = build_bucket_iterator(
        valid_dataset,
        batch_size=opt.valid_batch_size,
        is_train=False,
        device=device
    )

    # Run training
    trainer.run(train_iter, valid_iter, opt)
    

    for batch in train_iter:
        print(batch.source)
        print(batch.target)
        print(batch.sentences_scores)
        break

if __name__ == '__main__':
    train()




