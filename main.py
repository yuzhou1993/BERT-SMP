from Model import *
from utils import *
import pickle
import torch
from Layers import *
from torch import optim
import argparse
##note that:if you want to run this code on CSI300E,please change all 73 (the number of firm nodes on CSI100E) to 185 (the number of firm nodes on CSI300E)

parser = argparse.ArgumentParser()

parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. 1 True')
parser.add_argument('--optim', type=int, default='1',
                    help='0 SGD. 1 Adam')
parser.add_argument('--eval', type=int, default='1',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='400',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='50',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='5e-5',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='2',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='39',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='0.00098',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='20',
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.3',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='1.0',
                    help='rnn clip')
parser.add_argument('--save', type=bool, default=False,
                    help='save model')
parser.add_argument('--dataset', type=str, default='CSI100E',
                    help='dataset')
parser.add_argument('--log_name', type=str, default='log',
                    help='log_name')

##note that:if you want to run this code on CSI300E,you should change all the path of data

def load_dataset(device1, dataset):
    dataset_root = os.path.join('./data/', dataset)
    with open(os.path.join(dataset_root, 'x_num_standard.pkl'), 'rb') as handle:
        markets = pickle.load(handle)
    with open(os.path.join(dataset_root, 'y_1.pkl'), 'rb') as handle:
        y_load = pickle.load(handle)
    with open(os.path.join(dataset_root, 'x_newtext.pkl'), 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open(os.path.join(dataset_root, 'edge_new.pkl'), 'rb') as handle:
        edge_list=pickle.load(handle)
    with open(os.path.join(dataset_root, 'interactive.pkl'), 'rb') as handle:##the information of executives working in the company
        interactive_metric=pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, device=device1)
    x = x.to(torch.double)
    x_sentiment = torch.tensor(stock_sentiments, device=device1)
    x_sentiment = x_sentiment.to(torch.double)
    y = torch.tensor(y_load, device=device1).squeeze()
    y = (y>0).to(torch.long)
    inter_metric=torch.tensor(interactive_metric,device=device1)
    inter_metric=inter_metric.squeeze(2)
    inter_metric=inter_metric.transpose(0, 1)
    return x, y, x_sentiment,edge_list,inter_metric


def train(model, x_train, x_sentiment_train, y_train, edge_list, inter_metric, device1, rnn_length, NUM_STOCK):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 50
    for i in train_seq:
        output= model(x_train[i - rnn_length + 1: i + 1], x_sentiment_train[i - rnn_length + 1: i + 1], edge_list,inter_metric,device1)
        loss = criterion(output, y_train[i][:NUM_STOCK])
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / total_loss_count

def evaluate(model, x_eval, x_sentiment_eval, y_eval, edge_list, inter_metric, device1, rnn_length, NUM_STOCK):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    for i in seq:
        output= model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], edge_list,inter_metric,device1)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i][:NUM_STOCK].cpu().numpy())
    acc, auc, f1, mcc = metrics(trues, preds)
    return acc, auc, f1, mcc


if __name__=="__main__":
    args = parser.parse_args()

    # --- Logger (create early so all messages are recorded) ---
    logger = get_or_create_logger(args.log_name, './')

    # --- Device setup ---
    if args.device == 'cpu' or not torch.cuda.is_available():
        device1 = torch.device('cpu')
    else:
        device1 = torch.device(f'cuda:{args.device}')
    print(f"Using device: {device1}")

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

    # train-valid-test split (chronological, no overlap)
    x_train = x[:-100]
    x_eval = x[-100 - rnn_length:-50]
    x_test = x[-50 - rnn_length:]

    y_train = y[:-100]
    y_eval = y[-100 - rnn_length:-50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[:-100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length:-50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]

    # initialize
    best_model_file = "N/A"
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    model = GraphCNN(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS, out_c=2,
                     d_hidden=D_MARKET * 2, hidn_rnn=hidn_rnn, hid_c=hidn_att,
                     n_heads=N_heads, dropout=args.dropout, t_mix=t_mix)

    model.to(device1)
    model.to(torch.double)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train — only validate on eval set, NEVER on test set during training
    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train, x_sentiment_train, y_train,
                           edge_list, inter_metric, device1, rnn_length, NUM_STOCK)
        eval_acc, eval_auc, eval_f1, eval_mcc = evaluate(
            model, x_eval, x_sentiment_eval, y_eval,
            edge_list, inter_metric, device1, rnn_length, NUM_STOCK)

        eval_str = ("epoch {}, train_loss {:.4f}, eval_auc {:.4f}, "
                    "eval_acc {:.4f}, eval_f1 {:.4f}, eval_mcc {:.4f}").format(
            epoch, train_loss, eval_auc, eval_acc, eval_f1, eval_mcc)
        print(eval_str)
        logger.info(eval_str)

        if eval_auc > eval_epoch_best:
            eval_epoch_best = eval_auc
            eval_best_str = eval_str
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/eval_auc{:.4f}_acc{:.4f}_f1{:.4f}.pth".format(
                    eval_auc, eval_acc, eval_f1)
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch >= args.wait_epoch:
            print("saved_model_result:", eval_best_str)
            break
        epoch += 1

    if args.save and best_model_file != "N/A":
        model.load_state_dict(torch.load(best_model_file))
    test_acc, test_auc, test_f1, test_mcc = evaluate(
        model, x_test, x_sentiment_test, y_test,
        edge_list, inter_metric, device1, rnn_length, NUM_STOCK)
    test_str = "test_auc {:.4f}, test_acc {:.4f}, test_f1 {:.4f}, test_mcc {:.4f}".format(
        test_auc, test_acc, test_f1, test_mcc)
    print(test_str)

