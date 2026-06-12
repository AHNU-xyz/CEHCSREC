import torch
from torch import nn
import numpy as np
from src.layers.readout import *
from src.layers.hgnn import *
import torch.nn.functional as F


class CEHCSRec(nn.Module):
    def __init__(self, config):
        """
        初始化 CEHCSRec 类。

        :param config: 配置字典，包含模型的超参数和其他配置信息
        """
        super(CEHCSRec, self).__init__()
        self.item_num = config['item_num']
        self.hidden_size = config['hidden_size']

        # 嵌入层：将商品和类别嵌入到隐空间中
        self.embedding = nn.Embedding(config['item_num'] + config['category_num'], self.hidden_size, padding_idx=0)

        # 设备上的类别索引
        self.item_cates = torch.tensor(config['item_cates'], dtype=torch.long, device=config['device'])

        # 加权无损超图注意力卷积网络
        self.hgnn = HGAConv(self.hidden_size, config['step'], config['hg_dropout'])

        # 最后注意力机制，用于会话读出
        self.session_readout = LastAttention(self.hidden_size * 2)

        # 随机失活层，用于防止过拟合
        self.item_dropout = nn.Dropout(config['dropout'])

        # 损失函数：交叉熵损失
        self.loss = nn.CrossEntropyLoss()

        # 初始化模型参数
        self._reset_parameters()

        self.temperature = config.get('temperature', 0.5)
        self.contrastive_loss_fn = nn.CrossEntropyLoss()

    def _reset_parameters(self):
        """
        初始化模型参数，使用均匀分布。
        """
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def contrastive_loss(self, seq_output):
        """
        计算对比学习损失。

        :param seq_output: 序列输出 (b, hidden_size)
        :return: 对比学习损失
        """
        # 计算余弦相似度矩阵
        similarity_matrix = torch.matmul(seq_output, seq_output.T) / self.temperature
        labels = torch.arange(seq_output.size(0), device=seq_output.device)
        loss = self.contrastive_loss_fn(similarity_matrix, labels)
        return loss

    def nodes2items(self, nodes_hidden, alias):
        """
        根据别名获取节点的隐层表示。

        :param nodes_hidden: 节点的隐层表示
        :param alias: 别名
        :return: 序列的隐层表示
        """
        get = lambda i: nodes_hidden[i][alias[i]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias)).long()])
        return seq_hidden

    def get_scores(self, inputs):
        """
        获取模型的预测分数。

        :param inputs: 模型输入，包括商品序列、标签、节点、超图邻接矩阵、商品别名和类别别名
        :return: 预测分数
        """
        item_seq, label = inputs[0:2]  # (b, l) 商品序列 (b) 标签
        mask = item_seq.gt(0)
        nodes, hn_adj, alias_item, alias_cate = inputs[2:6]  # (b, node_num) (b, edge_num, node_num) (b, l) (b, l)

        # 嵌入并标准化节点
        nodes_hidden_in = self.embedding(nodes)  # (b, node_num, h)
        nodes_hidden_in = F.normalize(nodes_hidden_in, p=2, dim=-1)
        nodes_hidden_in = self.item_dropout(nodes_hidden_in)

        # 加权无损超图注意力卷积神经网络
        nodes_hidden, edges_hidden = self.hgnn(nodes_hidden_in, hn_adj)

        # 获取商品和类别的隐层表示
        item_seq_hidden = self.nodes2items(nodes_hidden, alias_item)
        cate_seq_hidden = self.nodes2items(nodes_hidden, alias_cate)

        # 将商品和类别的隐层表示连接起来
        seq_hidden = torch.cat([item_seq_hidden, cate_seq_hidden], dim=-1)
        seq_hidden = F.normalize(seq_hidden, p=2, dim=-1)
        seq_output = self.session_readout(seq_hidden, mask)

        # 获取商品和类别的嵌入
        item_emb = self.embedding.weight[:self.item_num, :]
        cates_emb = self.embedding(self.item_cates)
        item_emb = torch.cat([item_emb, cates_emb], dim=-1)

        # 标准化输出和嵌入
        seq_output = F.normalize(seq_output, p=2, dim=-1)
        item_emb = F.normalize(item_emb, p=2, dim=-1)

        # 计算分数
        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))
        scores = scores * 16
        return scores

    def get_session_embeddings(self, inputs):
        """
        提取会话嵌入。

        :param inputs: 模型输入，包括商品序列、标签、节点、超图邻接矩阵、商品别名和类别别名
        :return: 会话嵌入
        """
        nodes, hn_adj, alias_item, alias_cate = inputs[2:6]
        nodes_hidden_in = self.embedding(nodes)  # (b, node_num, h)
        nodes_hidden_in = F.normalize(nodes_hidden_in, p=2, dim=-1)
        nodes_hidden_in = self.item_dropout(nodes_hidden_in)
        nodes_hidden, edges_hidden = self.hgnn(nodes_hidden_in, hn_adj)
        session_embeddings = self.nodes2items(nodes_hidden, alias_item)
        return session_embeddings

    def get_category_embeddings(self, inputs):
        """
        提取类别嵌入。

        :param inputs: 模型输入，包括商品序列、标签、节点、超图邻接矩阵、商品别名和类别别名
        :return: 类别嵌入
        """
        nodes, hn_adj, alias_item, alias_cate = inputs[2:6]
        nodes_hidden_in = self.embedding(nodes)  # (b, node_num, h)
        nodes_hidden_in = F.normalize(nodes_hidden_in, p=2, dim=-1)
        nodes_hidden_in = self.item_dropout(nodes_hidden_in)
        nodes_hidden, edges_hidden = self.hgnn(nodes_hidden_in, hn_adj)
        category_embeddings = self.nodes2items(nodes_hidden, alias_cate)
        return category_embeddings

    def cal_loss(self, inputs):
        labels = inputs[1]
        scores = self.get_scores(inputs)
        scores = scores[:, 1:]

        # 主任务损失
        main_loss = self.loss(scores, labels - 1)

        # 对比学习损失
        seq_output = self.get_scores(inputs)  # 或在适当位置获取 seq_output
        contrastive_loss = self.contrastive_loss(seq_output)

        # 合并两种损失
        total_loss = main_loss + 0.1 * contrastive_loss  # 0.1 是对比学习损失的权重系数，可根据需要调整
        return total_loss

    def predict(self, inputs, phase):
        """
        预测函数。

        :param inputs: 模型输入，包括商品序列、标签、节点、超图邻接矩阵、商品别名和类别别名
        :param phase: 阶段（训练或测试）
        :return: 预测分数
        """
        scores = self.get_scores(inputs)
        scores = scores[:, 1:]

        return scores
