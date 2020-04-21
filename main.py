import torch
import numpy as np
import glob

from data.fieldsets.build_fieldsets import build_fieldset
from data.builders import build_training_datasets, build_test_dataset
from data.iterators import build_bucket_iterator
from data.utils import *
from data.corpus import Corpus
from trainers.utils import retrieve_trainer
from config import opt
from models import Model, BilstmPredictor, Estimator

from predictors.utils import setup_output_directory, configure_seed
from predictors.predictors import Predicter
from metrics.functions import *

def predict():
    
    # Setup
    output_dir = setup_output_directory(opt.pred_path, create=True)
    configure_seed(opt.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Model
    ModelClass = eval(opt.model_name)
    model = ModelClass.create_from_file(opt.model_path, opt)
    model = model.to(device)
    predicter = Predicter(model, opt)

    # Data
    fieldset = build_fieldset(opt)
    test_dataset = build_test_dataset(fieldset, opt, load_vocab=opt.model_path) # build vocabs from model?
    #test_dataset = build_test_dataset(fieldset, opt)
    #vocabs = fields_to_vocabs(test_dataset.fields)  # build vocab from test dataset?

    test_iter = build_bucket_iterator(
        test_dataset,
        batch_size=opt.test_batch_size,
        is_train=False,
        device=device
    )

    predictions = predicter.run(test_iter, batch_size=opt.test_batch_size)
    save_predicted_probabilities(opt.pred_path, predictions)

def evaluate():
    ## Ground-truth (z_mean)
    file_path = opt.paths['test']
    gt = []
    for filename in glob.glob(file_path + '*/*.tsv'):
        pdata = Corpus.read_tabular_file(filename)
        #print(len(pdata['z_mean']))
        gt.extend(pdata['z_mean'])
    gt = np.array(gt).astype(float)

    ## Predictions
    pred_file = opt.pred_path + 'scores'
    preds = np.array([line.strip() for line in open(pred_file)], dtype="float")

    ## Evaluate
    scores, ranks = eval_sentence_level(gt, preds)
    print_sentences_scoring_table(scores)
    print_sentences_ranking_table(ranks)

def train():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Data
    fieldset = build_fieldset(opt)
    
    if opt.model_name == 'Estimator':
        train_dataset, valid_dataset = build_training_datasets(fieldset, opt, split = 0.8, has_valid=False, load_vocab=opt.load_pred_source)
    else:
        train_dataset, valid_dataset = build_training_datasets(fieldset, opt, split = 0.8, has_valid=False)

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
    

    # for batch in train_iter:
    #     print(batch.source)
    #     print(batch.target)
    #     print(batch.sentences_scores)
    #     break

if __name__ == '__main__':
    train()
    #predict()
    #evaluate()




