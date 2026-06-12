from torch.utils.data import Dataset
from src.utils.data_utils import *


class ProcessDataset(Dataset):
    def __init__(self, config, data_dict):
        """
        初始化 ProcessDataset 类。

        :param config: 配置字典，包含数据集相关的参数和配置信息
        :param data_dict: 包含数据的字典
        """
        self.config = config
        self.data_dict = data_dict
        self.sample_data()
        self.max_seq_len = get_max_seq_len(data_dict, config)
        self.align_seq()
        self.window_size = self.config['window_size']

    def sample_data(self):
        """
        采样数据，根据配置字典中的 sample 参数截取样本数据。
        """
        self.data_dict = get_sample_data(data_dict=self.data_dict, config=self.config)

    def align_seq(self):
        """
        对齐序列，并处理类别序列。
        """
        self.data_dict = align_seq_category(data_dict=self.data_dict,
                                            max_seq_len=self.max_seq_len,
                                            config=self.config)

    def __getitem__(self, idx):
        """
        获取数据集中指定索引的数据。

        :param idx: 数据索引
        :return: 处理后的数据，包括商品序列、标签、节点、超图邻接矩阵、商品别名、类别别名和序列长度
        """
        item_seq = self.data_dict['item_seq'][idx]
        label = self.data_dict['label'][idx]
        category_seq = self.data_dict['category_seq'][idx]
        seq_len = self.data_dict['seq_len'][idx]

        nodes, hn_adj, alias_item, alias_cate = get_item_cate_hypergraph(
            item_seq, category_seq, seq_len, self.max_seq_len, self.config
        )

        # 转换为长整型张量
        long_tuple = item_seq, label, nodes, alias_item, alias_cate, seq_len
        float_tuple = hn_adj,
        item_seq, label, nodes, alias_item, alias_cate, seq_len = to_tensor_long(long_tuple)
        hn_adj, = to_tensor_float(float_tuple)

        # 返回处理后的样本
        sample = item_seq, label, nodes, hn_adj, alias_item, alias_cate, seq_len
        return sample

    def __len__(self):
        """
        返回数据集的长度。

        :return: 数据集的长度
        """
        return len(self.data_dict['item_seq'])
