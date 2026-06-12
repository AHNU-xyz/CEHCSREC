import torch
from torch import nn


# Weighted Lossless HyperGraph Attention Convolutional Neural Network
# 加权无损超图注意力卷积神经网络
class HGAConv(nn.Module):
    def __init__(self, hidden_dim, step, dropout=0.5):
        super(HGAConv, self).__init__()
        self.hidden_size = hidden_dim  # 隐藏层维度
        self.step = step
        self.dropout = nn.Dropout(dropout)

        # 定义两个线性层用于计算注意力权重
        self.q1 = nn.Linear(self.hidden_size + 1, 1, bias=False)
        self.q2 = nn.Linear(self.hidden_size + 1, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)  # LeakyReLU 激活函数

    def agg_cell(self, nodes_hidden, edge_hidden, hn_adj):
        """
        执行一次节点和超边之间的聚合操作。
        :param nodes_hidden: 节点隐藏表示 (batch_size, node_num, hidden_dim)
        :param edge_hidden: 超边隐藏表示 (batch_size, edge_num, hidden_dim)
        :param hn_adj: 超图邻接矩阵 (batch_size, edge_num, node_num)
        :return: 更新后的节点隐藏表示和超边隐藏表示
        """
        batch_size, edge_num, node_num = hn_adj.shape

        # ============= 节点到超边的聚合 =============
        edge_hidden_att = edge_hidden.unsqueeze(2).repeat(1, 1, node_num,
                                                          1)  # (batch_size, edge_num, node_num, hidden_dim)
        nodes_hidden_att = nodes_hidden.unsqueeze(1).repeat(1, edge_num, 1, 1)

        ele_hidden = torch.cat([edge_hidden_att * nodes_hidden_att, self.dropout(hn_adj.unsqueeze(-1))], dim=-1)

        alpha = self.leakyrelu(self.q1(ele_hidden)).squeeze(-1)  # (batch_size, edge_num, node_num)
        alpha.masked_fill_(hn_adj == 0, -1e10)
        alpha = torch.softmax(alpha, dim=-1)
        edge_hidden_new = torch.matmul(alpha, nodes_hidden)  # (batch_size, edge_num, hidden_dim)

        # ============= 超边到节点的聚合 =============
        hn_adj_t = hn_adj.transpose(1, 2)  # 转置超图邻接矩阵
        edge_hidden_attn = edge_hidden_new.unsqueeze(1).repeat(1, node_num, 1, 1)
        nodes_hidden_attn = nodes_hidden.unsqueeze(2).repeat(1, 1, edge_num, 1)

        ele_hidden = torch.cat([edge_hidden_attn * nodes_hidden_attn, self.dropout(hn_adj_t.unsqueeze(-1))], dim=-1)

        beta = self.leakyrelu(self.q2(ele_hidden)).squeeze(-1)  # (batch_size, node_num, edge_num)
        beta.masked_fill_(hn_adj_t == 0, -1e10)
        beta = torch.softmax(beta, dim=-1)

        nodes_hidden_new = torch.matmul(beta, edge_hidden_new)  # (batch_size, node_num, hidden_dim)

        return nodes_hidden_new, edge_hidden_new

    def forward(self, nodes_hidden, hn_adj):
        """
        前向传播函数，执行超图卷积操作。
        :param nodes_hidden: 节点隐藏表示 (batch_size, node_num, hidden_dim)
        :param hn_adj: 超图邻接矩阵 (batch_size, edge_num, node_num)
        :return: 更新后的节点隐藏表示和超边隐藏表示
        """
        edge_nodes_num = torch.sum(hn_adj, dim=-1).unsqueeze(-1)  # (batch_size, edge_num, 1)

        edge_nodes_num = torch.where(edge_nodes_num == 0., torch.ones_like(edge_nodes_num), edge_nodes_num)

        edge_hidden = torch.matmul(hn_adj, nodes_hidden)  # (batch_size, edge_num, hidden_dim)
        edge_hidden = edge_hidden / edge_nodes_num  # (batch_size, edge_num, hidden_dim)

        # 多次执行聚合操作
        for i in range(self.step):
            nodes_hidden, edge_hidden = self.agg_cell(nodes_hidden, edge_hidden, hn_adj)
        return nodes_hidden, edge_hidden
