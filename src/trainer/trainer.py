import os.path
import pickle
import time
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from src.utils.utils import early_stopping, print_result
from src.evaluator.evaluator import Evaluator


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j, neg_samples):
        sim_ij = self.cosine_similarity(z_i, z_j) / self.temperature
        sim_neg = torch.cat([self.cosine_similarity(z_i, neg).unsqueeze(1) for neg in neg_samples], dim=1) / self.temperature
        pos_loss = -torch.log(torch.exp(sim_ij) / (torch.exp(sim_ij) + torch.sum(torch.exp(sim_neg), dim=1)))
        return pos_loss.mean()

class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # 从配置文件中读取训练参数
        self.learning_rate = config['lr']
        self.l2 = config['l2']
        self.epochs = config['epoch_num']
        self.patience = config['patience']
        self.device = config['device']

        self.best_score = 0.0
        self.bad_step = 0
        self.topks = config['topk_list']
        self.evaluator = Evaluator(config)

        self.contrastive_loss_fn = InfoNCELoss()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._build_save_model()
        self.result_dict = {}
        self.contrastive_weight = config.get('contrastive_weight', 0.1)  # 对比学习的权重系数

    def _build_optimizer(self):
        # 构建优化器，使用Adam算法
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def _build_scheduler(self):
        # 构建学习率调度器，每隔lr_dc_step个epoch调整一次学习率
        scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                              step_size=self.config['lr_dc_step'], gamma=self.config['lr_dc'])
        return scheduler

    def _build_save_model(self):
        # 如果需要保存模型，创建保存路径
        if self.config['save']:
            save_model_path = '../save_model/{}'.format(self.config['model'])
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            self.save_model_file = save_model_path + '/{}_{}.txt'.format(self.config['dataset'],
                                                                         time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime()))
            self.save_model_file = self.save_model_file.replace(":", "-").replace(" ", "_")

    def _save_checkpoint(self, epoch):
        # 保存检查点，包含模型状态和优化器状态
        state = {
            'config': self.config,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, self.save_model_file)

    def train_epoch(self, tr_data):
        self.model.train()
        epoch_loss = 0.0
        contrastive_epoch_loss = 0.0  # 用于存储对比学习损失

        tr_data = tqdm(iterable=tr_data, total=len(tr_data), ncols=100)
        for batch_idx, batch_data in enumerate(tr_data):
            batch_data = [idata.to(self.device) for idata in batch_data[:-1]]
            self.optimizer.zero_grad()

            # 计算主任务的损失
            main_loss = self.model.cal_loss(batch_data)

            # 计算对比学习的损失
            seq_output = self.model.get_scores(batch_data)  # 确保模型能提供 `seq_output`
            contrastive_loss = self.model.contrastive_loss(seq_output)

            # 合并损失
            total_loss = main_loss + self.contrastive_weight * contrastive_loss
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += main_loss.item()
            contrastive_epoch_loss += contrastive_loss.item()

        epoch_loss /= len(tr_data)
        contrastive_epoch_loss /= len(tr_data)

        return epoch_loss, contrastive_epoch_loss

    def eval_epoch(self, eval_data, phase='normal'):
        # 评估一个epoch
        self.model.eval()
        hits, mrrs, ndcgs = [[] for _ in range(len(self.topks))], [[] for _ in range(len(self.topks))], [[] for _ in range(len(self.topks))]
        eval_data = tqdm(iterable=eval_data, total=len(eval_data), ncols=100)
        for batch_idx, batch_data in enumerate(eval_data):
            batch_data = [idata.to(self.device) for idata in batch_data[:-1]]
            scores = -self.model.predict(batch_data, phase)
            ranks = scores.argsort().argsort()
            labels = batch_data[1]

            batch_hits, batch_mrrs, batch_ndcgs = self.evaluator.eval(item_seqs=batch_data[0], labels=labels, ranks=ranks)
            for idx in range(len(self.topks)):
                hits[idx] += batch_hits[idx]
                mrrs[idx] += batch_mrrs[idx]
                ndcgs[idx] += batch_ndcgs[idx]

        hits = [np.mean(hit_list) * 100 for hit_list in hits]
        mrrs = [np.mean(mrr_list) * 100 for mrr_list in mrrs]
        ndcgs = [np.mean(ndcg_list) * 100 for ndcg_list in ndcgs]
        return hits, mrrs, ndcgs

    def fit(self, dataloaders):
        train_dl = dataloaders[0]
        test_dl = dataloaders[1]

        for epoch_idx in range(self.epochs):
            self.config['logger'].info('# === ' + 'Epoch: {}'.format(epoch_idx))

            # 训练阶段
            time_start = time.time()
            train_loss, contrastive_loss = self.train_epoch(train_dl)
            time_end = time.time()
            self.config['logger'].info('train time(s): {}'.format(round(time_end - time_start, 2)))
            self.config['logger'].info('train loss: {}'.format(round(train_loss, 4)))
            self.config['logger'].info('contrastive loss: {}'.format(round(contrastive_loss, 4)))

            # 评估阶段
            time_start = time.time()
            metric_results = self.eval_epoch(test_dl, phase='normal')
            time_end = time.time()
            self.config['logger'].info('test time(s): {}'.format(round(time_end - time_start, 2)))
            self.result_dict[epoch_idx] = metric_results
            hit_list, mrr_list, ndcg_list = metric_results
            print_result(self.config, metric_results, self.topks)

            # 早停机制
            curr_score = hit_list[-1]  # hit@top[max(k)]
            self.best_score, self.bad_step, update_flag, stop_flag = early_stopping(curr_score, self.best_score,
                                                                                    self.bad_step, self.patience)
            if update_flag:
                if self.config['save']:
                    self._save_checkpoint(epoch_idx)
                self.best_score = curr_score
            if stop_flag or epoch_idx == self.epochs - 1:
                self.config['logger'].info(
                    '# === Best Result from epoch {} , converge using {} epochs'.format(epoch_idx - self.patience,
                                                                                        epoch_idx - self.patience + 1))
                print_result(self.config, self.result_dict[epoch_idx - self.patience], self.topks)
                break
            self.scheduler.step()

