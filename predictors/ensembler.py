import torch
import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from joblib import dump, load
import glob

from data.fieldsets.build_fieldsets import build_fieldset
from data.builders import build_test_dataset
from data.iterators import build_bucket_iterator
from data.corpus import Corpus

from predictors.utils import configure_seed
from predictors.predictors import Predicter
from models import Model, BilstmPredictor, Estimator, TransformerPredictor

class Ensembler:

    def __init__(self, model_path_list, opt):

        configure_seed(opt.seed)
        self.opt = opt
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.predictors = [self.build_predictor(model_path) for model_path in model_path_list]
        self.num_files = len(model_path_list)
        self.model_path_list = model_path_list

        #self.clf = Ridge(alpha=.5)
        #self.clf = SGDRegressor(penalty='elasticnet')
        self.clf = AdaBoostRegressor(random_state=0, n_estimators=100)

    def build_predictor(self, model_path):

        # Model
        ModelClass = eval(self.opt.model_name)
        model = ModelClass.create_from_file(model_path, self.opt)
        model = model.to(self.device)
        predictor = Predicter(model, self.opt)

        return predictor

    def inference(self):

        ########## MODEL WITH SAME VOCABULARY ##########
        # # Data
        # fieldset = build_fieldset(self.opt)
        # test_dataset = build_test_dataset(fieldset, self.opt, load_vocab=self.opt.model_path)
        
        # print('Source vocabulary size: ', len(test_dataset.fields['source'].vocab.stoi))
        # print('Target vocabulary size: ', len(test_dataset.fields['target'].vocab.stoi))

        # test_iter = build_bucket_iterator(
        #     test_dataset,
        #     batch_size=self.opt.test_batch_size,
        #     is_train=False,
        #     device=self.device
        # )

        # predictions = np.zeros((len(test_iter), self.num_files))
        # for i, predictor in enumerate(self.predictors):
        #     prediction = predictor.run(test_iter, batch_size=self.opt.test_batch_size)
        #     predictions[:, i] = prediction['scores']
        
        # return predictions

        ########## MODEL WITH DIFFERENT VOCABULARY ##########
        predictions = []
        for i, (predictor, model_path) in enumerate(zip(self.predictors, self.model_path_list)):
            
            # Data
            fieldset = build_fieldset(self.opt)
            test_dataset = build_test_dataset(fieldset, self.opt, load_vocab=model_path)
            
            print('Source vocabulary size: ', len(test_dataset.fields['source'].vocab.stoi))
            print('Target vocabulary size: ', len(test_dataset.fields['target'].vocab.stoi))

            test_iter = build_bucket_iterator(
                test_dataset,
                batch_size=self.opt.test_batch_size,
                is_train=False,
                device=self.device
            )
            prediction = predictor.run(test_iter, batch_size=self.opt.test_batch_size)
            predictions.append(prediction['scores'])
        
        predictions = np.stack(predictions, axis=0).T

        return predictions
    
    def train(self, predictions):

        file_path = self.opt.paths[self.opt.dataset]
        Y = []
        for filename in glob.glob(file_path + self.opt.file):
            pdata = Corpus.read_tabular_file(filename)
            Y.extend(pdata['z_mean'])
        Y = np.array(Y).astype(float)

        self.clf.fit(predictions, Y)
        print('R square score: ', self.clf.score(predictions, Y))

        dump(self.clf, 'predictors/ensembler.joblib') 










        














        

