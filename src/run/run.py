import a  # 引入模块 a，但未在提供的代码中找到该模块的相关内容
import time
import torch
import argparse
from src.utils.utils import init_seed, get_item_num, get_dataset, get_dataloader, get_cate_num
from src.utils.log_utils import get_logger
from src.models.CEHCSRec import *
from src.trainer.trainer import Trainer

def run(opt):
    # 配置参数
    config = {
        'seed': 2022,  # 随机种子
        'topk_list': [10, 20],  # 推荐时考虑的topk列表
        'model': 'CLHHN',  # 使用的模型名称

        # 数据配置
        'dataset': opt.dataset,  # 数据集名称，可选值为 'tmall'、'diginetica'、'2019-oct'
        'sample': opt.sample,  # 数据采样大小，-1 表示全部数据，大于 0 表示采样大小
        'seq_reverse': True,  # 序列是否反转
        'item_num': get_item_num(opt.dataset),  # 商品数量
        'category_num': get_cate_num(opt.dataset),  # 类别数量
        'max_seq_len': opt.msl,  # 最大序列长度

        # 超图配置
        'window_size': 0,  # 窗口大小
        'step': 0,  # 步数
        'hg_dropout': 0.1,  # 超图 dropout

        # 训练配置
        'batch_size': 100,  # 批大小
        'hidden_size': 100,  # 隐藏层大小
        'dropout': 0.1,  # dropout 概率
        'lr': 0.001,  # 学习率
        'l2': 1e-5,  # L2 正则化系数
        'lr_dc': 0.1,  # 学习率衰减系数
        'lr_dc_step': 3,  # 学习率衰减步数
        'epoch_num': 100,  # 训练轮数
        'patience': 3,  # 早停耐心值
        'device': torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu"),  # 设备选择
        'save': False if opt.save == 0 else True,  # 是否保存模型
        'contrastive_temp': 0.5,   # 温度系数
        'contrastive_weight': 0.2   # 对比损失的权重
    }

    # 日志文件路径
    log_file = "../logs/log_{}_{}.txt".format(config['dataset'], time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    log_file = log_file.replace(":", "-").replace(" ", "_")
    config['logger'] = get_logger(log_file)  # 获取日志记录器

    # 根据数据集设置超图相关参数
    if config['dataset'] == 'tmall':
        config['window_size'] = 5
        config['step'] = 2
    elif config['dataset'] == 'diginetica':
        config['window_size'] = 4
        config['step'] = 2
    elif config['dataset'] == '2019-oct':
        config['window_size'] = 6
        config['step'] = 2
    else:
        config['window_size'] = 5
        config['step'] = 2

    logger = config['logger']
    logger.info(config)  # 记录配置信息

    init_seed(config['seed'])  # 初始化随机种子

    # 加载数据集
    train_ds, test_ds, maps = get_dataset(config)
    # 获取数据加载器
    dataloaders = get_dataloader(config, (train_ds, test_ds))

    # 构建商品与类别的映射关系
    item_cate_map = maps[0]
    item_cates = [item_cate_map[item_id] + config['item_num'] for item_id in range(config['item_num'])]
    config['item_cates'] = item_cates

    # 创建模型实例并移动到设备上
    model = CEHCSRec(config).to(config['device'])
    config['logger'].info(model)

    # 创建训练器
    trainer = Trainer(config, model)
    # 进行模型训练
    trainer.fit(dataloaders)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tmall')
    parser.add_argument('--msl', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--save', type=int, default=0)
    opt = parser.parse_args()

    # 运行主程序
    run(opt)
