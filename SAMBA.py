from __future__ import annotations
import json
from types import SimpleNamespace

from einops import rearrange, repeat, einsum
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class SAMBA(nn.Module):
    def __init__(self, args, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = args
        self.hidden = hidden
        self.inp = inp
        self.out = out
        self.embed = embed
        self.cheb_k = cheb_k
        self.num_nodes = args.vocab_size

        # Mamba 模块
        self.mam1 = Mamba(args, hid=hidden)
        self.mam2 = Mamba(args, hid=hidden)

        # 节点嵌入用于构造动态邻接矩阵
        self.adj = nn.Parameter(torch.randn(self.num_nodes, embed))  # e.g., [5, 32]
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))  # [32, 3, 73, 1]
        nn.init.xavier_uniform_(self.weights_pool)

        self.bias = nn.Parameter(torch.FloatTensor(out))
        nn.init.constant_(self.bias, 0.1)

        # 输出层（注意：输入为5节点，输出为单个概率）
        self.final_proj = nn.Sequential(nn.Linear(self.num_nodes, 1), nn.Sigmoid())

    def gaussian_kernel_graph(self, E_A, x, gamma=1.0):
        x_mean = torch.mean(x, dim=1)  # [batch, N * features]
        x_mean = x_mean.reshape(x_mean.size(0), self.num_nodes, -1)  # [batch, N, features]
        x_mean = torch.mean(x_mean, dim=0)  # [N, features]

        E_A_exp = E_A.unsqueeze(1)  # [N, 1, embed]
        E_A_T_exp = E_A.unsqueeze(0)  # [1, N, embed]
        dist = torch.sum((E_A_exp - E_A_T_exp) ** 2, dim=2)  # [N, N]
        ADJ = torch.exp(-gamma * dist)
        ADJ = F.softmax(ADJ, dim=1)  # [N, N]
        return ADJ

    def forward(self, input_ids):
        # input_ids: [B, N, seq, F]
        batch_size, num_nodes, seq_len, num_features = input_ids.shape
        x = input_ids.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # [B, seq, N*F]

        # 时序建模
        xx = self.mam1(x)  # [B, seq, N*F]
        xx = xx.reshape(batch_size, seq_len, num_nodes, num_features).permute(0, 2, 1, 3)  # [B,N,seq,F]
        xx_flat = xx.reshape(batch_size, num_nodes, -1)  # [B, N, seq*F]

        # 动态图卷积
        ADJ = self.gaussian_kernel_graph(self.adj, xx_flat)  # [N, N]
        I = torch.eye(num_nodes).to(input_ids.device)
        support_set = [I, ADJ]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)  # [K, N, N]

        x_g = torch.einsum("knm,bmc->bknc", supports, xx_flat)  # [B, K, N, C]
        x_g = x_g.reshape(batch_size, self.cheb_k, num_nodes, seq_len, num_features)
        x_g = x_g.permute(0, 2, 1, 3, 4)  # [B, N, K, seq, F]

        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)  # [N, K, F, out]
        out = torch.einsum('bnktf,nkfo->bno', x_g, weights) + self.bias  # [B, N, out]
        out = self.final_proj(out.permute(0, 2, 1).to(input_ids.device)).squeeze(1)  # [B, 1]
        out = torch.sigmoid(out)
        return out.squeeze(-1)



class Mamba(nn.Module):
    def __init__(self, args, hid):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.nl=args.n_layer

        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        self.layers2 = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])

        #self.layers3 = nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),AVWGCN(args.seq_in,args.seq_in,2,args.d_model)) for _ in range(args.n_layer)])

        #self.layers3=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),AVWGCN(args.seq_in,args.seq_in,2,args.d_model)) for _ in range(args.n_layer)])

        #self.layers4=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),gconv(args.seq_in,hid,2,10,args.d_model),nn.ReLU(),gconv(hid,args.seq_in,2,10,args.d_model)) for _ in range(args.n_layer)])
        self.lin=nn.ModuleList([nn.Sequential(nn.LayerNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in)) for _ in range(args.n_layer-2)]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))])
        #self.lin2=nn.ModuleList([nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in)) for _ in range(args.n_layer-2)]+[nn.Sequential(RMSNorm(args.seq_in),nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))])
        self.norm_f = nn.LayerNorm(args.d_model)
        #self.lm_head = nn.Linear(args.d_model, args.vocab_size)
        self.proj=nn.Sequential(nn.Linear(args.seq_in,hid),nn.ReLU(),nn.Linear(hid,args.seq_in))
        self.nnl=nn.LayerNorm(args.vocab_size)
        #self.proj=nn.Linear(2*ModelArgs.vocab_size, ModelArgs.vocab_size)
        #self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
                                                     # See "Weight Tying" paper


    def forward(self, input_ids):
        x = self.embedding(input_ids)
        residual = x
        x1 = x
        x2 = x
        for i in range(self.nl):
            x1 = self.layers[i](x1)
            x2 = self.layers2[i](x2.flip([1]))
            x = x1 + x2.flip([1]) + x
            x = self.lin[i](x.permute(0,2,1)).permute(0,2,1) + x
            x1 = x
            x2 = x
        x = self.norm_f(x + residual)
        return x


    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))


        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)

        config_data = load_config_hf(pretrained_model_name)
        args = SimpleNamespace(**{
            "d_model": config_data['d_model'],
            "n_layer": config_data['n_layer'],
            "vocab_size": config_data['vocab_size'],
            "seq_in": config_data.get('seq_in', 20),  # 添加必要字段，若无则设默认
            "d_inner": config_data.get('d_inner', 128),
            "d_state": config_data.get('d_state', 16),
            "dt_rank": config_data.get('dt_rank', 8),
            "bias": config_data.get('bias', True),
            "conv_bias": config_data.get('conv_bias', True)
        })
        model = Mamba(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model





class ResidualBlock(nn.Module):
    def __init__(self, args):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = nn.LayerNorm(args.d_model)


    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output


class gconv(nn.Module):
    def __init__(self, inp, hid,embed,cheb_k,n):
        super(gconv, self).__init__()

        self.node_num=n

        self.inp=inp

        self.cheb_k=cheb_k

        self.adj=nn.Parameter(torch.randn(n,embed), requires_grad=True)



        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, hid))

        self.bias_pool = nn.Parameter(torch.FloatTensor(embed,hid))

    def forward(self, x):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        ADJ=F.softmax(F.relu(torch.mm(self.adj, self.adj.transpose(0, 1))), dim=1)
        support_set = [torch.eye(self.node_num).cuda(),ADJ]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])

        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.adj, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        out_6 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias   #B,N,D_OUT

        return out_6

class AVWGCN(nn.Module):
    def __init__(self, dim_in, hid, cheb_k,n):
        super(AVWGCN, self).__init__()
        self.node_num=n
        self.inp=dim_in
        self.cheb_k = cheb_k
        self.node_embeddings = nn.Parameter(torch.randn(n,dim_in,dim_in), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(cheb_k,n,dim_in, hid))

        self.bias_pool = nn.Parameter(torch.FloatTensor(n, hid))

    def forward(self, x):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        supports = F.softmax(F.relu(self.node_embeddings), dim=2)
        I=torch.eye(self.inp).cuda()
        I2=I[None,:,:].repeat(x.size(1),1,1)
        support_set = [I2, supports]
        supports = torch.stack(support_set, dim=0)

        #N, dim_out
        x_g = torch.einsum("bnc,kncm->bknm", x, supports)      #B, cheb_k, N, dim_in
        #x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bknm,knmo->bno', x_g, self.weights_pool) + self.bias_pool     #b, N, dim_out
        return x_gconv



class MambaBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = nn.Linear(args.vocab_size, args.d_model)
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        # 修正 A_log 的初始化
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, args.d_state + 1, dtype=torch.float32)
                .repeat(args.d_inner, 1)  # 形状 [d_inner=128, d_state=16]
            )
        )
        self.D = nn.Parameter(torch.ones(args.d_inner))

    def forward(self, x):
        (b, l, d) = x.shape  # d = args.d_model=365
        x_and_res = self.in_proj(x)  # 输入维度 365 → 输出 d_inner*2
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y


    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output