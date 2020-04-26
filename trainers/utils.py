import torch
from torch import optim
from more_itertools import collapse
import numpy as np
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
        model = ModelClass.create_from_file(opt.model_path, opt, vocabs=vocabs)
    else:
        model = ModelClass.from_options(vocabs=vocabs, opt=opt, PreModelClass=opt.pre_model_name)
    
    model = model.to(device)
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print (name, param.data)

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
        opt,
        log_interval=opt.log_interval,
        scheduler=scheduler
    )
    return trainer


# For metrics
def precision(tp, fp, fn):
    if tp + fp > 0:
        return tp / (tp + fp)
    return 0

def recall(tp, fp, fn):
    if tp + fn > 0:
        return tp / (tp + fn)
    return 0

def fscore(tp, fp, fn):
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    return 0

def precision_recall_fscore_support(hat_y, y, labels=None):
    n_classes = len(labels) if labels else None
    cnfm = confusion_matrix(hat_y, y, n_classes)

    if n_classes is None:
        n_classes = cnfm.shape[0]

    scores = np.zeros((n_classes, 4))
    for class_id in range(n_classes):
        scores[class_id] = scores_for_class(class_id, cnfm)
    return scores.T.tolist()

def confusion_matrix(hat_y, y, n_classes=None):
    hat_y = np.array(list(collapse(hat_y)))
    y = np.array(list(collapse(y)))

    if n_classes is None:
        classes = np.unique(np.union1d(hat_y, y))
        n_classes = len(classes)

    cnfm = np.zeros((n_classes, n_classes))
    for j in range(y.shape[0]):
        cnfm[y[j], hat_y[j]] += 1
    return cnfm


def scores_for_class(class_index, cnfm):
    tp = cnfm[class_index, class_index]
    fp = cnfm[:, class_index].sum() - tp
    fn = cnfm[class_index, :].sum() - tp
    tn = cnfm.sum() - tp - fp - fn

    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    f1 = fscore(tp, fp, fn)
    support = tp + tn
    return p, r, f1, support













