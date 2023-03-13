'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import argparse
import os

import albumentations as A
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.text_folder_dataset import TextFolderDataset

from utils import (Transform, build_model, load_checkpoint, make_dataset,
                   read_py_config)


def main():
    # parsing arguments
    parser = argparse.ArgumentParser(description='antispoofing training')
    parser.add_argument('--draw_graph', default=False, type=bool, required=False,
                        help='whether or not to draw graphics')
    parser.add_argument('--GPU', default=0, type=int, required=False,
                        help='specify which GPU to use')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='if you want to eval model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # reading config and manage device
    path_to_config = args.config
    config = read_py_config(path_to_config)
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'

    # building model
    model = build_model(config, device, strict=True, mode='eval')
    model.to(device)
    if config.data_parallel.use_parallel:
        model = nn.DataParallel(model, **config.data_parallel.parallel_params)

    # load snapshot
    path_to_experiment = os.path.join(config.checkpoint.experiment_path, config.checkpoint.snapshot_name)
    epoch_of_checkpoint = load_checkpoint(path_to_experiment, model, map_location=device, optimizer=None, strict=False)
    
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, torch.ones([1,3,224,224],dtype=torch.float32, device=torch.device('cuda:0')))
    print(flops.total())

#    from ptflops import get_model_complexity_info
   
#    with torch.cuda.device(0):     
#        macs, params = get_model_complexity_info(model.module, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
#        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    exit(0)    
    # preprocessing, making dataset and loader
    
    normalize = A.Normalize(**config.img_norm_cfg)
    test_transform = A.Compose([
                                A.Resize(**config.resize, interpolation=cv.INTER_CUBIC),
                                normalize
                               ])
    test_transform = Transform(val=test_transform)
    test_dataset = TextFolderDataset(root_folder="./datasets/FAS-CVPR2023",
                     data_folder="dev/norm_crop",
                     txt_filename="Dev.txt", transform=test_transform)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=8)
    

    
    # computing metrics
    final_str = evaluate(model, test_loader,config, device, set_name='dev')
    
    with open("predictions_CVPR.txt","w") as f_pred:
        f_pred.write(final_str)
    
    print ("Done!")
    
def evaluate(model, loader, config, device, set_name='dev'):
    ''' evaluating AUC, EER, BPCER, APCER, ACER on given data loader and model '''
    model.eval()
    proba_accum = np.array([])
    fnames_acum = np.array([])
    
    
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (image, target) in loop:
        if config.test_steps == i:
            break
        image = image.to(device)

        with torch.no_grad():
            features = model(image)
            if config.data_parallel.use_parallel:
                model1 = model.module
            else:
                model1 = model
            output = model1.make_logits(features, all=False)
            if isinstance(output, tuple):
                output = output[0]

            y_pred = output.argmax(dim=1).detach().cpu().numpy()
            
            
            if config.loss.amsoftmax.margin_type in ('cos', 'arcos'):
                output *= config.loss.amsoftmax.s
            if config.loss.loss_type == 'soft_triple':
                output *= config.loss.soft_triple.s
            positive_probabilities = F.softmax(output, dim=-1)[:,1].cpu().numpy()
        proba_accum = np.concatenate((proba_accum, positive_probabilities))
        target_current = target.cpu().numpy()
        filenames = f"{set_name}/"+ loader.dataset.data['filename'][target_current].to_numpy()
        
        fnames_acum = np.concatenate((fnames_acum, filenames))
        #target_accum = np.concatenate((target_accum, y_true))
        #print(f"{positive_probabilities}")
        #print(f"{loader.dataset.data['filename'][ids_acum].to_numpy()}")
        
      
    to_return = np.array([fnames_acum, proba_accum]).T.tolist()
    final_str = "\n".join([f"{e[0]} {e[1]}" for e in to_return])
    
    return final_str

def plot_roc_curve(fpr, tpr, config):
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.00])
    plt.plot(fpr, tpr, lw=3, label="ROC curve (area= {:0.2f})".format(auc(fpr, tpr)))
    plt.xlabel('FPR', fontsize=16)
    plt.ylabel('TPR', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0,1],[0,1], lw=3, linestyle='--', color='navy')
    plt.savefig(config.curves.det_curve)

def det_curve(fps,fns, eer, config):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    fig,ax = plt.subplots(figsize=(8,8))
    plt.plot(fps,fns, label=f"DET curve, EER%={round(eer*100, 3)}")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('FAR', fontsize=16)
    plt.ylabel('FRR', fontsize=16)
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.xticks(rotation=45)
    plt.axis([0.001,1,0.001,1])
    plt.title('DET curve', fontsize=20)
    plt.legend(loc='upper right', fontsize=16)
    fig.savefig(config.curves.det_curve)

if __name__ == "__main__":
    main()
