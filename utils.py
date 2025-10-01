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
    # 合并所有batch的结果
    trues = np.concatenate(trues, axis=0).flatten()  # [n_samples]
    preds = np.concatenate(preds, axis=0)  # [n_samples, n_classes]

    # 获取预测标签（必须是离散值0/1）
    pred_labels = preds.argmax(axis=-1)  # 关键修改！取概率最大的类别作为预测标签

    # 计算各项指标
    acc = accuracy_score(trues, pred_labels)
    auc = roc_auc_score(trues, preds[:, 1])  # 假设是二分类任务，取正类概率
    f1 = f1_score(trues, pred_labels, average='macro')  # 二分类需指定average
    mcc = matthews_corrcoef(trues, pred_labels)

    true_classes = np.hstack(trues)
    # print("2",pred_classes)
    # 统计preds中正样本和负样本的数量
    TP = np.sum((true_classes == 1) & (pred_labels == 1))  # 真正例
    FN = np.sum((true_classes == 1) & (pred_labels == 0))  # 假负例
    FP = np.sum((true_classes == 0) & (pred_labels == 1))  # 假正例
    TN = np.sum((true_classes == 0) & (pred_labels == 0))  # 真负例

    # # 打印结果
    # print("真实为正样本且预测为正样本的个数 (TP): ", TP)  # 真实为正样本且预测为正样本的个数
    # print("真实为正样本但预测为负样本的个数 (FN): ", FN)  # 真实为正样本但预测为负样本的个数
    # print("真实为负样本但预测为正样本的个数 (FP): ", FP)  # 真实为负样本但预测为正样本的个数
    # print("真实为负样本且预测为负样本的个数 (TN): ", TN)  # 真实为负样本且预测为负样本的个数


    return acc, auc, f1, mcc

    # return acc, auc

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

