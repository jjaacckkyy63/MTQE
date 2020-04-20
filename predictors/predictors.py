import torch
from tqdm import tqdm
import logging
from collections import defaultdict

from models.model import Model

class Predicter:

    def __init__(self, model, opt, fields=None):
        """Class to load a model for inference.
        Args:
          model (kiwi.models.Model): A trained QE model
          fields (dict[str: Field]): A dict mapping field names to strings.
            For online prediction.
        """

        self.model = model
        self.fields = fields
        #self.logger = self.get_logger(opt)

    def run(self, iterator, batch_size=1):
        self.model.eval()
        predictions = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(iterator, total=len(iterator)):
                model_out = self.model(batch)
                pred = {}
                pred['scores'] = model_out['scores'].tolist()
                for key, values in pred.items():
                    if isinstance(values, list):
                        predictions[key] += values
                    else:
                        predictions[key].append(values)
        return dict(predictions)


    def get_logger(self, opt):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        
        path = opt.pred_path+'{}.log'.format(opt.model_name)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

        return logger
