import torch
from tqdm import tqdm
import logging

from models.model import Model

class Trainer:

    def __init__(self, model, optimizer, opt, log_interval=100, scheduler=None):
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
        self.logger = self.get_logger(opt)

    
    def run(self, train_iter, valid_iter, opt):
        self.logger.info(' ')
        self.logger.info('Start Logging...')
        
        for epoch in range(opt.epochs + 1):
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

                if self._step % opt.log_interval == 0:
                    self.logger.info("Current Training Loss at step {}: {}".format(self._step, loss_dict['loss']))
            
                if opt.checkpoint_validation_steps and self._step % opt.checkpoint_validation_steps == 0:
                    # validation
                    print('\n======= Validation Start =======')
                    validation_loss = 0
                    self.model.eval()
                    with torch.no_grad():
                        for batch in valid_iter:
                            model_out = self.model(batch)
                            loss_dict = self.model.loss(model_out, batch)
                            val_outputs = dict(loss=loss_dict, model_out=model_out)

                            validation_loss += loss_dict['loss']
                        
                        validation_loss /= len(valid_iter)
                        self.logger.info(" ====== Current Validation Loss at step {}: {} ====== ".format(self._step, validation_loss))

                    self.model.train()
                    print('======= Validation Done =======')
            
            if epoch % opt.save_checkpoint_interval == 0:
                self.model.save(opt.checkpoint_path+'{}_{}.pth'.format(opt.model_name, epoch))
    
    def get_logger(self, opt):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        
        path = opt.checkpoint_path+'{}.log'.format(opt.model_name)
        handler = logging.FileHandler(path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

        return logger
            

            
    




