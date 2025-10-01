from Layers import *
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import scipy.sparse as sp
from transformers import BertForSequenceClassification, BertConfig, BertModel


class GraphCNN(nn.Module):
    def __init__(self, num_stock, d_market, d_news, out_c, d_hidden, hidn_rnn, hid_c, dropout, alpha=0.2, alpha1=0.0054,
                 t_mix=1, n_layeres=2, n_heads=1):  ##alpha1 denotes the normalized threshold
        super(GraphCNN, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        self.num_stock = num_stock
        self.gcs = nn.ModuleList()
        self.X2Os = Graph_Linear(num_stock, hidn_rnn * 2, out_c, bias=True)

        bert_config = BertConfig.from_pretrained('./bert/')
        bert_config.num_attention_heads = bert_config.hidden_size // 6
        # bert_config.num_hidden_layers = 1
        self.bert_classification = BertForSequenceClassification(bert_config)

        cross_bert_config = BertConfig.from_pretrained('./bert/')
        cross_bert_config.num_attention_heads = cross_bert_config.hidden_size // 13
        cross_bert_config.add_cross_attention = True
        cross_bert_config.position_embedding_type = 'relative'
        cross_bert_config._attn_implementation = 'eager'
        cross_bert_config.num_hidden_layers = 1
        # cross_bert_config.hidden_size = 39
        cross_bert_config.num_stock = num_stock
        cross_bert_config.num_executives = 275
        self.bert_model = BertModel(cross_bert_config)

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)


    def forward(self, x_market, x_news, edge_list, inter_metric, device1):

        x_s = torch.cat([x_market, x_news], dim=-1)
        x_s = x_s.transpose(0, 1).double()
        x_s = self.bert_classification(inputs_embeds=x_s)
        inter_metric = inter_metric.double()
        if inter_metric.shape[1] != x_s.shape[0]:
            inter_metric = inter_metric.transpose(0, 1)
        x = (inter_metric @ x_s)  # Generate the initial features of executive nodes

        x_ss = x_s.unsqueeze(0)
        x = x.unsqueeze(0)

        edge_index = []
        for eg in ['R', 'C', 'CL']:
            edge_index += edge_list[eg]

        p_num = x.shape[1]
        p_head_mask = torch.zeros([p_num, p_num])
        for e in edge_index:
            p_head_mask[e[0]][e[1]] = 1.0
            p_head_mask[e[1]][e[0]] = 1.0

        p_head_mask = p_head_mask.to(device1)

        edge_index=[]
        for eg in ['FS','FE']:
            edge_index += edge_list[eg]
        c_p_head_mask = torch.zeros(self.num_stock, p_num)
        p_c_head_mask = torch.zeros(p_num, self.num_stock)
        for e in edge_index:
            c_p_head_mask[e[0]][e[1]] = 1.0
            p_c_head_mask[e[1]][e[0]] = 1.0

        # c_p_head_mask = c_p_head_mask.to(device1)
        # p_c_head_mask = p_c_head_mask.to(device1)

        outputs = self.bert_model(
            inputs_embeds=x_ss,
            encoder_hidden_states=x,
            head_mask=p_head_mask,
            c_p_head_mask=c_p_head_mask,
            p_c_head_mask=p_c_head_mask,
        )

        x_c = outputs[0].squeeze()

        x_0 = torch.cat([x_s, x_c], dim=1)
        x_0 = F.elu(self.X2Os(x_0))
        out = F.log_softmax(x_0, dim=1)

        return out