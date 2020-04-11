import torch
from tqdm import tqdm
from models.model import Model

class Trainer:

    def __init__(self, model, optimizer, log_interval=100, scheduler=None):
        """
        Args:
          model: A Model to train
          optimizer: An optimizer
          checkpointer: A Checkpointer object
          log_interval: Log train stats every /n/ batches. Default 100
          scheduler: A learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._step = 0
        self._epoch = 0

    
    def run(self, train_iter, valid_iter, epochs=50):
        
        for epoch in range(self._epoch + 1, epochs + 1):
            print('Epoch {} of {}'.format(epoch, epochs))
            
            # train
            self.model.train()

            for batch in tqdm(
                train_iter,
                total=len(train_iter),
                desc='Batches',
                unit=' batches',
                ncols=80,
            ):
                self._step += 1
                self.model.zero_grad()
                model_out = self.model(batch)
                loss_dict = self.model.loss(model_out, batch)
                loss_dict['loss'].backward()
                self.optimizer.step()
                
                train_outputs = dict(loss=loss_dict, model_out=model_out)

            # validation
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(
                    valid_iter,
                    total=len(valid_iter),
                    desc='Batches',
                    unit=' batches',
                    ncols=80,
                ):
                    model_out = self.model(batch)
                    loss_dict = self.model.loss(model_out, batch)
                    val_outputs = dict(loss=loss_dict, model_out=model_out)
            
            self.model.train()
                
            self._epoch += 1
            
    




