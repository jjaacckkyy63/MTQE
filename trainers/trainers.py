import torch
from tqdm import tqdm
import logging

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
        self.logger = self.get_logger()

    
    def run(self, train_iter, valid_iter, opt):
        
        for epoch in range(self._epoch + 1, opt.epochs + 1):
            print('Epoch {} of {}'.format(epoch, opt.epochs))
            self.logger.info('Epoch {} of {}'.format(epoch, opt.epochs))
            
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
                self.logger.info("Current Training Loss at step {}: {}".format(self._step, loss_dict['loss']))

                if self._step % opt.log_interval == 0:
                    print()
            
                if opt.checkpoint_validation_steps and self._step % opt.checkpoint_validation_steps == 0:
                    # validation
                    validation_loss = 0
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

                            validation_loss += loss_dict['loss']
                        
                        validation_loss /= len(valid_iter)*opt.valid_batch_size
                        self.logger.info("====== Current Validation Loss at step {}: {} ======".format(self._step, loss_dict['loss']))

                    self.model.train()
            
            if opt.save_checkpoint_interval == 0:
                self.model.save(opt.checkpoint_path+'{}_{}'.format(self.model.__name__, epoch))

            self._epoch += 1
    
    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        
        handler = logging.FileHandler('output.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

        return logger
            

            
    




