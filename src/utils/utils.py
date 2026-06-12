import pickle
import random
from src.data.data import *
from torch.utils.data import DataLoader


def init_seed(seed):
    """
    初始化随机种子以确保实验的可重复性。
    :param seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_item_num(dataset_name):
    """
    获取指定数据集中的商品数量。
    :param dataset_name: 数据集名称
    :return: 商品数量
    """
    dataset2item_num = {'diginetica': 43097 + 1,
                        'tmall': 40727 + 1,
                        '2019-oct': 27480 + 1
                        }
    return dataset2item_num[dataset_name]


def get_cate_num(dataset_name):
    """
    获取指定数据集中的类别数量。
    :param dataset_name: 数据集名称
    :return: 类别数量
    """
    dataset2cate_num = {'diginetica': 995 + 1,
                        'tmall': 711 + 1,
                        '2019-oct': 448 + 1
                        }
    return dataset2cate_num[dataset_name]


def get_dataset(config):
    """
    加载并返回训练和测试数据集。
    :param config: 配置字典，包含数据集名称等信息
    :return: 训练数据集，测试数据集和映射关系
    """
    config['logger'].info("get_dataset...")
    data_path = '../../datasets/' + config['dataset'] + '/idata.txt'
    with open(data_path, 'rb') as f:
        data_dicts = pickle.load(f)
    train_dict, test_dict, maps = data_dicts
    train_ds = ProcessDataset(config, train_dict)
    test_ds = ProcessDataset(config, test_dict)
    return train_ds, test_ds, maps


def get_dataloader(config, datasets):
    """
    创建并返回训练和测试数据加载器。
    :param config: 配置字典，包含批量大小等信息
    :param datasets: 包含训练数据集和测试数据集的元组
    :return: 训练数据加载器，测试数据加载器
    """
    config['logger'].info("get_dataloader...")
    train_ds, test_ds = datasets
    drop_last = False
    if torch.cuda.is_available():
        train_dl = DataLoader(dataset=train_ds, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
        test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True, drop_last=drop_last)
    else:
        train_dl = DataLoader(dataset=train_ds, batch_size=config['batch_size'],
                              shuffle=False, num_workers=1, pin_memory=False, drop_last=drop_last)
        test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'],
                             shuffle=False, num_workers=1, pin_memory=False, drop_last=drop_last)
    return train_dl, test_dl


def early_stopping(curr_result, best_result, cur_step, patience):
    """
    早停机制，用于在验证集表现不佳时提前停止训练。
    :param curr_result: 当前结果
    :param best_result: 最佳结果
    :param cur_step: 当前步数
    :param patience: 容忍度
    :return: 更新后的最佳结果，步数，更新标志和停止标志
    """
    update_flag, stop_flag = False, False
    if curr_result > best_result:
        cur_step = 0
        best_result = curr_result
        update_flag = True
    else:
        cur_step += 1
        if cur_step >= patience:
            stop_flag = True
    return best_result, cur_step, update_flag, stop_flag


def print_result(config, metric_results, topks):
    """
    打印和记录评估结果。
    :param config: 配置字典，包含日志记录器
    :param metric_results: 评估指标结果
    :param topks: top-k 的列表
    """
    hits, mrrs, ndcgs = metric_results
    width, style = '{:<15}', '%.3f'
    # print('-' * 110)
    # print('|', end='')
    for idx, topk in enumerate(topks):
        # print('|',
        #       width.format(style % hits[idx]),
        #       width.format(style % mrrs[idx]),
        #       width.format(style % ndcgs[idx]),
        #       '|', end=''
        #       )
        curr_result = '  |' + \
                      width.format(style % hits[idx]) + \
                      width.format(style % mrrs[idx]) + \
                      width.format(style % ndcgs[idx]) + \
                      '|'
        config['logger'].info(curr_result)
    # print('|')
    # print('||', width.format(style % his_dcg),
    #       width.format(style % his_mrr),
    #       )
    # print('-' * 110)
