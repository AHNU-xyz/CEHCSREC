import torch
import torch.nn as nn

class LastAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LastAttention, self).__init__()
        self.hidden_size = hidden_dim  # 隐藏层维度
        self.pos_size = int(hidden_dim / 2)  # 位置嵌入维度
        self.position_embedding = nn.Embedding(100, self.pos_size)  # 位置嵌入
        self.w_hp = nn.Linear(self.hidden_size + self.pos_size, self.hidden_size, bias=False)  # 用于计算 z 的线性层
        self.w1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # 用于计算注意力权重的线性层
        self.w2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # 用于计算注意力权重的线性层
        self.q = nn.Linear(self.hidden_size, 1, bias=False)  # 最终注意力权重的线性层

    def forward(self, seq_hidden, mask):
        """
        前向传播函数，计算注意力权重并生成会话表示。
        :param seq_hidden: 序列的隐藏表示 (batch_size, seq_len, hidden_dim)
        :param mask: 掩码，用于标记有效的序列元素 (batch_size, seq_len)
        :return: 加权后的序列表示 (batch_size, hidden_dim)
        """
        batch_size, seq_len = mask.shape
        mask = mask.float().unsqueeze(-1)  # 扩展掩码以便与隐藏表示相乘

        # 获取位置嵌入
        pos_emb = self.position_embedding.weight[:seq_len]  # (seq_len, pos_size)
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, seq_len, pos_size)

        # 获取序列的第一个隐藏表示
        h_t = seq_hidden[:, 0]  # (batch_size, hidden_dim)
        h_t = h_t.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)

        # 计算 z 向量
        z = torch.tanh(self.w_hp(torch.cat([seq_hidden, pos_emb], dim=-1)))  # (batch_size, seq_len, hidden_dim)

        # 计算注意力权重 beta
        beta = self.q(torch.sigmoid(self.w1(z) + self.w2(h_t)))  # (batch_size, seq_len, 1)

        # 计算加权后的序列表示
        seq_output = torch.sum(beta * seq_hidden * mask, dim=1)  # (batch_size, hidden_dim)

        return seq_output
