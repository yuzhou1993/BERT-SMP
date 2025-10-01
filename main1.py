import numpy as np
# from anaconda_project.internal.cli.command_commands import add_command
# from holoviews.operation import gradient
# from sympy.abc import epsilon

from Model import *
from utils import *
import pickle
import torch
from Layers import *
from torch import optim
import argparse
from transformers import get_linear_schedule_with_warmup
import pandas as pd

##note that:if you want to run this code on CSI300E,please change all 73 (the number of firm nodes on CSI100E) to 185 (the number of firm nodes on CSI300E)

parser = argparse.ArgumentParser()

parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='200',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='8e-5',  #   CSI100：8e-4,   CSI300:8e-5
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='2', #原始2
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='78',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='2e-8',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='20',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.5',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='1.0',
                    help='rnn clip')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')
parser.add_argument('--dataset', type=str, default='CSI300E',
                    help='dataset')
parser.add_argument('--log_name', type=str, default='CSI300EBERT',
                    help='log_name')




def load_dataset(device1, dataset):
    dataset_root = os.path.join('./data/', dataset)
    with open(os.path.join(dataset_root, 'x_num_standard.pkl'), 'rb') as handle:
        markets = pickle.load(handle)
    with open(os.path.join(dataset_root, 'y_1.pkl'), 'rb') as handle:
        y_load = pickle.load(handle)
    with open(os.path.join(dataset_root, 'x_newtext.pkl'), 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open(os.path.join(dataset_root, 'edge_new.pkl'), 'rb') as handle:
        edge_list = pickle.load(handle)
    with open(os.path.join(dataset_root, 'interactive.pkl'),
              'rb') as handle:  ##the information of executives working in the company
        interactive_metric = pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=device1)
    x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=device1)
    x_sentiment.to(torch.double)
    y = torch.tensor(y_load, device=device1).squeeze()
    y = (y > 0).to(torch.long)
    inter_metric = torch.tensor(interactive_metric, device=device1)
    inter_metric = inter_metric.squeeze(2)
    inter_metric = inter_metric.transpose(0, 1)
    return x, y, x_sentiment, edge_list, inter_metric




def train(model, x_train, x_sentiment_train, y_train, edge_list, inter_metric, device1):
    model.train()

    # x_train = x_train[:84]

    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    # print("Shuffled train_seq samples:", train_seq[:10])
    adv_total_loss_val = 0
    total_loss_val = 0
    adv_total_loss_count = 0
    total_loss_count = 0
    epsilon = 0.000091
    batch_train = 35  #CSI300：35    CSI100 52
    # fgm = FGM(model)
    for i in train_seq:
        # optimizer.zero_grad()
        x_input = x_train[i - rnn_length + 1: i + 1]
        x_sentiment_input = x_sentiment_train[i - rnn_length + 1: i + 1]
        output = model(x_input, x_sentiment_input, edge_list, inter_metric, device1)  #只有技术指标
        loss = criterion(output, y_train[i][:NUM_STOCK])
        loss.backward()
        total_loss_val += loss.item()
        total_loss_count += 1

        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

    return total_loss_val / total_loss_count


def evaluate(model, x_eval, x_sentiment_eval, y_eval, edge_list, device1):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    epsilon = 0.01

    for i in seq:

        x_input = x_eval[i - rnn_length + 1: i + 1]
        x_sentiment_input = x_sentiment_eval[i - rnn_length + 1: i + 1]
        output = model(x_input, x_sentiment_input, edge_list, inter_metric, device1)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i][:NUM_STOCK].cpu().numpy())
    acc, auc, f1, mcc = metrics(trues, preds)
    return acc, auc, f1, mcc


if __name__ == "__main__":
    args = parser.parse_args()
    device1 = "cuda:" + args.device
    device1 = device1
    print(device1)
    criterion = torch.nn.NLLLoss()
    set_seed(1021)
    # load dataset
    print("loading dataset")
    x, y, x_sentiment, edge_list, inter_metric = load_dataset(device1, args.dataset)
    # hyper-parameters
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    MAX_EPOCH = args.max_epoch
    hidn_rnn = args.hidn_rnn
    N_heads = args.heads_att
    hidn_att = args.hidn_att
    lr = args.lr
    rnn_length = args.rnn_length
    t_mix = 0
    edge_list = edge_list

    # train-valid-test split
    x_train = x[: -100]
    x_eval = x[-100 - rnn_length: -50]
    x_test = x[-50 - rnn_length:]

    y_train = y[: -100]
    y_eval = y[-100 - rnn_length: -50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[: -100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length: -50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]

    ## initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    test_epoch_best = 0

    model = GraphCNN(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS, out_c=2,
                     d_hidden=D_MARKET + D_NEWS, hidn_rnn=hidn_rnn, hid_c=hidn_att, n_heads=N_heads,
                     dropout=args.dropout, t_mix=t_mix)

    model.cuda(device=device1)

    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_constraint)


    logger = get_or_create_logger(args.log_name, './')
    print(args.log_name)

    # train
    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train, x_sentiment_train, y_train, edge_list, inter_metric, device1)
        eval_acc, eval_auc, f1, mcc, = evaluate(model, x_eval, x_sentiment_eval, y_eval, edge_list, device1)
        test_acc, test_auc, test_f1, test_mcc = evaluate(model, x_test, x_sentiment_test, y_test, edge_list, device1)
        print("Epoch:", epoch, "LR:", optimizer.param_groups[0]["lr"])
        eval_str1 = "epoch{},train_loss{:.4f}, auc{:.4f},acc{:.4f},f1{:.4f},mcc{:.4f}".format(epoch, train_loss,
                                                                                              test_auc, test_acc,
                                                                                              test_f1, test_mcc)
        logger.info(eval_str1)



        if test_auc > test_epoch_best:
            test_epoch_best = test_auc
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f},test_acc{:.4f}".format(
                epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels" + "/" + args.log_name
                print("save目录", best_model_file)
                torch.save(model.state_dict(), best_model_file)
        else:
            # wait_epoch += 1
            wait_epoch += 0

        if wait_epoch >= 50:
            print("saved_model_result:", eval_str1)
            break
        epoch += 1
    print("saved_model_result:", eval_best_str)