import torch
from torch import optim
from trainers.trainers import Trainer

# Define function for training

def optimizer_class(name):
    if name == 'sgd':
        OptimizerClass = optim.SGD
    elif name == 'adagrad':
        OptimizerClass = optim.Adagrad
    elif name == 'adadelta':
        OptimizerClass = optim.Adadelta
    elif name == 'adam':
        OptimizerClass = optim.Adam
    elif name == 'sparseadam':
        OptimizerClass = optim.SparseAdam
    else:
        raise RuntimeError("Invalid optim method: " + name)
    return OptimizerClass

def retrieve_trainer(ModelClass, opt, vocabs, device):

    # Build model
    if opt.model_path:
        model = ModelClass.create_from_file(opt.model_path)
    else:
        model = ModelClass.from_options(vocabs=vocabs, opt=opt)
    
    model = model.to(device)

    # Build optimizer
    Optimizer = optimizer_class(opt.optimizer)
    optimizer = Optimizer(model.parameters(), lr = opt.lr)
    
    scheduler = None
    if 0.0 < opt.learning_rate_decay < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=opt.learning_rate_decay,
            patience=opt.learning_rate_decay_start,
            verbose=True,
            mode="max",
        )

    # Build trainer
    trainer = Trainer(
        model,
        optimizer,
        log_interval=opt.log_interval,
        scheduler=scheduler,
    )
    return trainer











