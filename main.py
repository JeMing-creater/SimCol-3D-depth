import os
import yaml
import time
import monai
import torch
import numpy as np
import monai.losses
import monai.metrics
from torch import nn
from typing import Dict
from objprint import objstr
from datetime import datetime
from easydict import EasyDict
from accelerate import Accelerator
from timm.optim import optim_factory

from src.models.FCBFormer_D import FCBFormer_D
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.dataloader import get_dataloaders, get_testloader
from src.utils import Logger, load_pretrain_model, same_seeds


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          metrics: Dict[str, monai.metrics.CumulativeIterationMetric], accelerator: Accelerator, epoch: int, step: int):
    model.train()
    for i, data in enumerate(train_loader):
        # divide data
        img = data[0]
        map = data[1]
        # output
        out = model(img)
        # compute loss
        total_loss = 0
        for name in loss_functions:
            loss = loss_functions[name](out, map)
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            total_loss += loss
        # backward
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        # print and log
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(
            f'Epoch [{epoch+1}/{config.trainer.num_epochs}][{i + 1}/{len(train_loader)}] Training Loss:{total_loss}',
            flush=True)
        step += 1
    # change learning rate
    scheduler.step(epoch)
    return step


def val_one_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, loss_functions: Dict[str, torch.nn.modules.loss._Loss], epoch: int):
    model.eval()
    t = time.time()
    perf_accumulator = []
    for i, data in enumerate(val_loader):
        # divide data
        img = data[0]
        map = data[1]
        # output
        out = model(img)
        perf = loss_functions['mse_loss'](out, map).item()
        for i in range(len(out)):
            perf_accumulator.append(perf)
        if i + 1 < len(val_loader):
            log = "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    len(perf_accumulator),
                    len(val_loader.dataset),
                    100.0 * len(perf_accumulator) / len(val_loader.dataset),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            accelerator.print(log, flush=True)
            accelerator.log(log)
        else:
            log = "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    len(perf_accumulator),
                    len(val_loader.dataset),
                    100.0 * (i + 1) / len(val_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            accelerator.print(log, flush=True)
            accelerator.log(log)
    return np.mean(perf_accumulator), np.std(perf_accumulator)


if __name__ == "__main__":
    # base setting
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now()).replace('-','_').replace(' ','_').replace('.','_').replace(':','_')
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))
    
    # dataset
    accelerator.print('load dataset...')
    train_loader, val_loader = get_dataloaders(config.dataset.root, config.dataset.img_size, config.dataset.batch_size, config.dataset.num_workers)

    # model
    model = FCBFormer_D(checkpoint_root=config.models.FCBFormer_D.checkpoint_root)
    
    # optimizer
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=config.trainer.betas)
    # scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    
    # loss functions
    loss_functions = {
        'mse_loss': nn.MSELoss(),
        # 'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    # metrics
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=True,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False),
        'L1': monai.metrics
    }
    
    # set in the devices
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)
    
    accelerator.print("Start Training！")
    step = 0
    starting_epoch = 0
    best_mean = 10000
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        step = train_one_epoch(model, loss_functions, train_loader,
                     optimizer, scheduler, metrics, accelerator, epoch, step)
        test_measure_mean, test_measure_std = val_one_epoch(model, val_loader, loss_functions, epoch)
        
        if test_measure_mean < best_mean:
            accelerator.print("Save Best Model!")
            best_mean = test_measure_mean
            accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/new/")
        
        accelerator.print("Now best mean loss: {:.6f}".format(best_mean))
        
        accelerator.print('Checkout....')
        accelerator.save_state(output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint")
        torch.save({'epoch': epoch, 'best_acc': best_mean},
                    f'{os.getcwd()}/model_store/{config.finetune.checkpoint}/checkpoint/epoch.pth.tar')
        accelerator.print('Checkout Over!')