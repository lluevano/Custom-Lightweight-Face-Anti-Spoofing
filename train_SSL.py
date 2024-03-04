import argparse

import albumentations as A
import cv2 as cv
import torch
import torch.nn as nn

from trainer import Trainer
from utils import (Transform, build_criterion, build_model, make_dataset,
                   make_loader, make_weights, read_py_config, save_checkpoint)
from models import BYOL
import torchvision.transforms as T
import torchvision.transforms.v2
from tqdm import tqdm

import os

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--GPU', type=int, default=0, help='specify which gpu to use')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Configuration file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='if you want to train model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # manage device, arguments, reading config
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    if config.data_parallel.use_parallel:
        device = f'cuda:{config.data_parallel.parallel_params.output_device}'
    if config.multi_task_learning and config.dataset != 'celeba_spoof':
        raise NotImplementedError(
            'Note, that multi task learning is avaliable for celeba_spoof only. '
            'Please, switch it off in config file'
            )
    # launch training, validation, testing
    train(config, device)

def train(config, device):
    normalize = T.v2.Normalize(**config.img_norm_cfg)
    default_transform = A.Compose([
                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                #normalize
                ])
    train_transform_blur = T.Compose([
                            #A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            T.v2.ToDtype(torch.uint8),
                            T.v2.Resize((config.resize.height, config.resize.width), interpolation=T.InterpolationMode.BICUBIC),
                            T.v2.RandomHorizontalFlip(p=0.5),
                            T.v2.ToDtype(torch.float32),
                            #A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                            #                                    intensity=(0.2, 0.5), p=0.2),
                            T.ColorJitter(brightness=0.2, contrast=0.2),
                            T.v2.GaussianBlur(kernel_size=5),
                            normalize
                            ])
    train_transform_no_blur = T.Compose([
                            T.v2.ToDtype(torch.uint8),
                            #A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                            T.v2.Resize((config.resize.height, config.resize.width), interpolation=T.InterpolationMode.BICUBIC),
                            T.v2.RandomHorizontalFlip(p=0.5),
                            T.v2.ToDtype(torch.float32),
                            #A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                            #                                    intensity=(0.2, 0.5), p=0.2),
                            T.ColorJitter(brightness=0.2, contrast=0.2),
                            #A.augmentations.MotionBlur(blur_limit=5, p=0.2),
                            normalize
                            ])
    train_transform = Transform(train_spoof=default_transform,
                                train_real=default_transform, val=None)
    train_dataset, val_dataset, test_dataset = make_dataset(config, train_transform, train_transform)
    train_loader, val_loader, test_loader = make_loader(train_dataset, val_dataset,
                                                        test_dataset, config)

    original_model = build_model(config, device=device, strict=False, mode='train')
    learner = BYOL(
                    original_model,
                    image_size = config.resize.height,
                    hidden_layer = -1,
                    #projection_size = 256,           # the projection size
                    #projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
                    #moving_average_decay = 0.99, 
                    use_momentum = False,       # turn off momentum in the target encoder
                    augment_fn = train_transform_blur,
                    augment_fn2 = train_transform_no_blur,
                )
    original_model.to(device)
    learner.to(device)
    if config.data_parallel.use_parallel:
        model = torch.nn.DataParallel(learner, **config.data_parallel.parallel_params)
    else:
        model = learner
    opt = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **config.scheduler)

    for epoch in range(config.epochs.start_epoch, config.epochs.max_epoch):
        if epoch != config.epochs.start_epoch:
            scheduler.step()
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for i, (input_, target) in loop:
            input_.to(device)
            target.to(device)
            #print(device)
            #print(input_.shape)
            loss = model(input_)
            loss = torch.mean(loss)
            #print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

            max_epochs = config.epochs.max_epoch
            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(avr_loss = loss,
                             lr=opt.param_groups[0]['lr'])
    
    if config.data_parallel.use_parallel:
        original_model = torch.nn.DataParallel(original_model, **config.data_parallel.parallel_params)
    checkpoint = {'state_dict': original_model.state_dict(),
                   'optimizer': opt.state_dict(), 'epoch': config.epochs.max_epoch}

    path_to_checkpoint = os.path.join(config.checkpoint.experiment_path, config.checkpoint.snapshot_name)
    save_checkpoint(checkpoint, f'{path_to_checkpoint}')

if __name__=='__main__':
    main()
