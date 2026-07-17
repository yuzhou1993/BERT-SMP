from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,matthews_corrcoef
import os, logging

def get_or_create_logger(logger_name=None, log_dir=None):
    logger = logging.getLogger(logger_name)

    # check whether handler exists
    if len(logger.handlers) > 0:
        return logger

    # set default logging level
    logger.setLevel(logging.DEBUG)

    # define formatters
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    # define and add handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, logger_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(trues, preds):
    trues = np.concatenate(trues, axis=0).flatten()  # [n_samples]
    preds = np.concatenate(preds, axis=0)  # [n_samples, n_classes]
    pred_labels = preds.argmax(axis=-1)  

    acc = accuracy_score(trues, pred_labels)
    auc = roc_auc_score(trues, preds[:, 1])  
    f1 = f1_score(trues, pred_labels, average='average')  
    mcc = matthews_corrcoef(trues, pred_labels)
    true_classes = np.hstack(trues)
    TP = np.sum((true_classes == 1) & (pred_labels == 1)) 
    FN = np.sum((true_classes == 1) & (pred_labels == 0))  
    FP = np.sum((true_classes == 0) & (pred_labels == 1))  
    TN = np.sum((true_classes == 0) & (pred_labels == 0))  
    
    return acc, auc, f1, mcc

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

