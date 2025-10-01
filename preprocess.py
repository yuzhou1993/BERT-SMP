import pickle, torch
import numpy as np

device1 = 'cpu'

with open('./data/CSI100E/x_num_standard.pkl', 'rb') as handle:
    markets = pickle.load(handle)
with open('./data/CSI100E/y_1.pkl', 'rb') as handle:
    y_load = pickle.load(handle)
with open('./data/CSI100E/x_newtext.pkl', 'rb') as handle:
    stock_sentiments = pickle.load(handle)
with open('./data/CSI100E/edge_new.pkl', 'rb') as handle:
    edge_list=pickle.load(handle)
with open('./data/CSI100E/interactive.pkl', 'rb') as handle:##the information of executives working in the company
    interactive_metric=pickle.load(handle)

markets = markets.astype(np.float64)
x = torch.tensor(markets, device=device1)
x.to(torch.double)
x_sentiment = torch.tensor(stock_sentiments, device=device1)
x_sentiment.to(torch.double)
y = torch.tensor(y_load, device=device1).squeeze()
y = (y>0).to(torch.long)
inter_metric=torch.tensor(interactive_metric,device=device1)
inter_metric=inter_metric.squeeze(2)
inter_metric=inter_metric.transpose(0, 1)

print(x.shape)  #(516,73,5)
print(x_sentiment.shape)   #(516,73,3)
print(y.shape)  #(516,236)
print(inter_metric.shape)  #(163, 73)

 
# 打开.pkl文件并读取
with open('./data/CSI100E/edge_new.pkl', 'rb') as f:
    data = pickle.load(f)    # I B S SC FS FE C R CL
                                # I 544
                                # B 196
                                # S 14
                                # SC 54
                                # FS 1
                                # FE 166
                                # C 676
                                # R 4
                                # CL 1906
    # for r in data:
    #     print(r, len(data[r]))
    print(data)

with open('./data/CSI100E/interactive.pkl', 'rb') as f:
    data = pickle.load(f)    # (73, 163, 1)
    print(data)

with open('./data/CSI100E/x_newtext.pkl', 'rb') as f:
    data = pickle.load(f)    # (516, 73, 3) 天 公司 向量
    print(data)

with open('./data/CSI100E/x_num_standard.pkl', 'rb') as f:
    data = pickle.load(f)    # (516, 73, 5)
    print(data)

with open('./data/CSI100E/y_1.pkl', 'rb') as f:
    data = pickle.load(f)    # (516, 236, 1)
    print(data)