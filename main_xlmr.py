import torch
import numpy as np
import glob
import logging
from tqdm import tqdm
from joblib import dump, load
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import pickle

from data.fieldsets.build_fieldsets import build_fieldset
from data.builders import build_training_datasets, build_test_dataset
from data.iterators import build_bucket_iterator
from data.utils import *
from data.corpus import Corpus
from trainers.utils import retrieve_trainer
from config import opt
from models import Model, XLMRPredictor, XLMREstimator

from predictors.utils import setup_output_directory, configure_seed
from predictors import Predicter, Ensembler
from metrics.functions import *

from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM


###################### DATA ######################
class Data(Dataset):
    
    def __init__(self, dataset, data_path, done=True):
        
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        if done:
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = self.parse_data(opt.paths[dataset])
            with open(data_path, 'wb') as f:
                pickle.dump(self.data, f)
        
        self.all_data, self.score = self.data
        self.vocab = self.tokenizer.get_vocab()
        
    
    def parse_data(self, file_path):

        data = []
        for filename in glob.glob(file_path + opt.file):
            print('Process file done: ', filename)
            pdata = Corpus.read_tabular_file(filename)
            for source, target, score in zip(pdata['original'], pdata['translation'], pdata['z_mean']):
                source_id = self.tokenizer.encode(source, add_special_tokens=True)
                target_id = self.tokenizer.encode(target, add_special_tokens=True)
                source_id_p = self.tokenizer.encode(source, add_special_tokens=False)
                target_id_p = self.tokenizer.encode(target, add_special_tokens=False)
                pair_id = self.tokenizer.build_inputs_with_special_tokens(source_id_p, target_id_p)
                data.append((source_id, target_id, pair_id, float(score)))
        
        source_data, target_data, all_data, score = zip(*data)
        
        source_id_t = [torch.tensor(sent) for sent in source_data]
        target_id_t = [torch.tensor(sent) for sent in target_data]
        all_id_t = [torch.tensor(sent) for sent in all_data]
        score_t = torch.tensor(list(score))

        source_id_t = pad_sequence(source_id_t, batch_first=True, padding_value=1)
        target_id_t = pad_sequence(target_id_t, batch_first=True, padding_value=1)
        all_id_t = pad_sequence(all_id_t, batch_first=True, padding_value=1)

        return all_id_t, score_t
            
    def __getitem__(self, index):
        return self.all_data[index], self.score[index]

    def __len__(self):
        return len(self.all_data)

def get_dataloader(dataset, data_path, done=True):
    dataset = Data(dataset, data_path, done)
    return DataLoader(dataset, batch_size=opt.train_batch_size, shuffle=False)
###################### DATA ######################

def predict():
    
    opt.out_embeddings_size = 768
    opt.train_batch_size = 16
    opt.valid_batch_size = 16

    # Setup
    output_dir = setup_output_directory(opt.pred_path, create=True)
    configure_seed(opt.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    configure_seed(opt.seed)

    test_iter = get_dataloader('test', 'raw_data/processed/test.pkl', False)

    # Model
    ModelClass = eval(opt.model_name)
    model = ModelClass.create_from_file(opt.model_path, opt)
    model = model.to(device)
    predicter = Predicter(model, opt)

    # Run prediction
    predictions = predicter.run(test_iter, batch_size=opt.test_batch_size)
    save_predicted_probabilities(opt.pred_path, predictions)

    

def train():
    
    opt.out_embeddings_size = 768
    opt.train_batch_size = 16
    opt.valid_batch_size = 16
    opt.lr = 2e-3

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    configure_seed(opt.seed)

    train_iter = get_dataloader('train', 'raw_data/processed/train.pkl', True)
    valid_iter = get_dataloader('valid', 'raw_data/processed/valid.pkl', True)
    print(len(train_iter))
    print(len(valid_iter))

    # Trainer
    ModelClass = eval(opt.model_name)
    trainer = retrieve_trainer(ModelClass, opt, train_iter.dataset.vocab, device)

    # Run training
    trainer.run(train_iter, valid_iter, opt)
    # model = XLMRPredictor(train_iter.dataset.vocab, opt)
    # for batch in train_iter:
    #     pair_hidden = model(batch)
    #     break

    # print(pair_hidden[0].shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Put arguments for train, predict, or evaluate')
    parser.add_argument('-m', '--mode', dest='mode', default='train', help='Input the mode: train, predict or evaluate')
    parser.add_argument('-d', '--dataset', dest='dataset', default='test', help='train, valid, test dataset')
    parser.add_argument('-f', '--file', dest='file', default='en-*/*.tsv', help='file to use')

    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'predict':
        opt.dataset = args.dataset
        opt.file = args.file
        predict()
    else:
        pass

    

