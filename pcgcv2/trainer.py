import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME

from loss import get_bce, get_bits, get_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter
from data_loader import dense_to_sparse, sparse_to_dense
from torch import nn
import pandas as pd

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, inputs, target):
        return torch.mean(torch.abs(inputs - target))

class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = { 'bpp':[],'sum_loss':[]}
        # self.record_set = {'mse':[], 'bpp':[],'sum_loss':[], 'metrics':[]}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))   
        # return zero
        for k in self.record_set.keys():
            self.record_set[k] = []

        return 

    @torch.no_grad()
    def test(self, dataloader, excel_test, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        mse_loss = nn.MSELoss()
        mae_loss = MAELoss()
        record = { 'bpp':[],'sum_loss':[]}
        # for _, (coords, feats) in enumerate(tqdm(dataloader)):
        for _, x in enumerate(tqdm(dataloader)):
            # data
            # x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
            # # Forward.
            ori = x
            x = dense_to_sparse(x)
            out_set = self.model(x, training=False)

            # bce loss
            # bce, bce_list = 0, []
            # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            #     curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
            #     bce += curr_bce
            #     bce_list.append(curr_bce.item())

            # mse loss
            out = out_set['out']
            mse = mae_loss(out.F, x.F)

            # bpp loss
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            # sum_loss = self.config.alpha * mse     #stage1
            sum_loss = self.config.alpha * mse + self.config.beta * bpp #stage2

            metrics = []
            # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            #     metrics.append(get_metrics(out_cls, ground_truth))
            # for ori, rec in zip(ori, rec):
            #     metrics.append(get_metrics(ori, rec))
            # record
            # self.record_set['bce'].append(bce.item())
            # self.record_set['bces'].append(bce_list)
            # self.record_set['mse'].append(mse.item())
            self.record_set['bpp'].append(bpp.item())
            self.record_set['sum_loss'].append(sum_loss.item())  #stage1
            # self.record_set['sum_loss'].append(mse.item() + bpp.item())
            # self.record_set['metrics'].append(metrics)


            torch.cuda.empty_cache()# empty cache.

        rec_bpp = sum(self.record_set['bpp']) / len(self.record_set['bpp'])
        rec_mse = sum(self.record_set['sum_loss']) / len(self.record_set['sum_loss'])

        self.record(main_tag=main_tag, global_step=self.epoch)

        record['bpp'] = rec_bpp
        record['sum_loss'] = rec_mse

        return record

    def train(self, dataloader, excel_train):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        mse_loss = nn.MSELoss()
        mae_loss = MAELoss()

        record = { 'bpp':[],'sum_loss':[]}
        start_time = time.time()
        # for batch_step, (coords, feats) in enumerate(tqdm(dataloader)):
        for batch_step, x in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            # x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
            # if x.shape[0] > 6e5: continue
            # forward
            ori = x
            x = dense_to_sparse(x)
            out_set = self.model(x, training=True)

            # bce loss
            # bce, bce_list = 0, []
            # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
            #     curr_bce = get_bce(out_cls, ground_truth)/float(out_cls.__len__())
            #     # curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
            #     bce += curr_bce
            #     bce_list.append(curr_bce.item())


            # mse loss
            out = out_set['out']
            mse = mae_loss(out.F, x.F)

            # bpp loss
            bpp = get_bits(out_set['likelihood'])/float(x.__len__())

            # sum_loss = self.config.alpha * mse #stage1
            sum_loss = self.config.alpha * mse + self.config.beta * bpp #stage2


            # backward & optimize
            sum_loss.backward()
            self.optimizer.step()
            # metric & record
            with torch.no_grad():
                metrics = []
                # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                #     metrics.append(get_metrics(out_cls, ground_truth))for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                # for ori, rec in zip(ori, rec):
                #     metrics.append(get_metrics(ori, rec))
                # self.record_set['bce'].append(bce.item())
                # self.record_set['bces'].append(bce_list)
                # self.record_set['mse'].append(mse.item())
                self.record_set['bpp'].append(bpp.item())
                self.record_set['sum_loss'].append(sum_loss.item()) #stage1
                # self.record_set['sum_loss'].append(mse.item() + bpp.item())
                # self.record_set['metrics'].append(metrics)
                if (time.time() - start_time) > self.config.check_time*60:
                    # self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                    self.save_model()
                    start_time = time.time()

            torch.cuda.empty_cache()# empty cache.

        rec_bpp = sum(self.record_set['bpp'])/len(self.record_set['bpp'])
        rec_mse = sum(self.record_set['sum_loss'])/len(self.record_set['sum_loss'])

        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1

        record['bpp'] = rec_bpp
        record['sum_loss'] = rec_mse

        return record
