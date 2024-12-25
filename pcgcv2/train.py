import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader, ImageFolder
from pcc_model import PCCModel
from trainer import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", default='/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/Hou/dataset/kitti_64_2048_layer1')
    # parser.add_argument("--dataset", default='/media/student/e15ac0f9-6b08-41c6-b970-5f45828d563b/Hou/Ford')
    parser.add_argument("--dataset_num", type=int, default=2e4)

    parser.add_argument("--alpha", type=float, default=100000., help="weights for distortion.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    # parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--init_ckpt", default='/home/student/Desktop/Salmon/PCGCv2-master_0414/ckpts/240517_alpha10000_kitti_mae_stage2_with_ANF/epoch_689.pth')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='240604_alpha100000_kitti_stage2_ANF', help="prefix of checkpoints/logger, etc.")

    parser.add_argument("--patch-size", type=int, nargs=2, default=(64, 2048), help="Size of the patches to be cropped (default: %(default)s)")
 
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, check_time):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.check_time=check_time


if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('./logs', args.prefix), 
                            ckptdir=os.path.join('./ckpts', args.prefix), 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            lr=args.lr, 
                            check_time=args.check_time)
    # model
    model = PCCModel()
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, transform = train_transforms, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=("cuda"),
    )
    test_dataset = ImageFolder(args.dataset, transform = test_transforms, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=("cuda"),
    )

    # filedirs = sorted(glob.glob(args.dataset+'*.h5'))[:int(args.dataset_num)]
    # train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    # train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    # test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
    # test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

    import pandas as pd
    excel_train = os.path.join('./ckpts', args.prefix, 'train_records.xlsx')
    excel_test = os.path.join('./ckpts', args.prefix, 'test_records.xlsx')
    rec_train = []
    rec_test = []
    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr
        record_train = trainer.train(train_dataloader, excel_train)
        record_test = trainer.test(test_dataloader, excel_test, 'Test')
        rec_train.append(record_train)
        rec_test.append(record_test)
    df_record_train = pd.DataFrame(rec_train)
    df_record_train.to_excel(excel_train, index=False)
    df_record_test = pd.DataFrame(rec_test)
    df_record_test.to_excel(excel_test, index=False)



