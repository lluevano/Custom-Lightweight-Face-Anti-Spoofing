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
import cv2
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

#  Code taken from https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py

from skimage import transform as trans

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)
    
def estimate_norm(lmk, image_size=112,mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size%112==0 or image_size%128==0
    if image_size%112==0:
        ratio = float(image_size)/112.0
        diff_x = 0
    else:
        ratio = float(image_size)/128.0
        diff_x = 8.0*ratio
    dst = arcface_dst * ratio
    dst[:,0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M

def norm_crop(img, landmark, image_size=224, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
    



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
    parser.add_argument('--img_path', type=str, required=True,
                        help='Single image path')
    args = parser.parse_args()
    
    
    img = cv2.imread(args.img_path)
    if img is None:
        raise f"Could not open image {args['img_path']}"
        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img,(img.shape[1]-(img.shape[1]%32), img.shape[0]-(img.shape[0]%32)),cv2.INTER_CUBIC)
    
    app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(img.shape[0], img.shape[1]))
    lmks = app.get(img, max_num=1)[0]['kps']
        
    img = norm_crop(img, lmks)
    
    img = (img-127.5)/128.0
    
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    
    img_tensor = torch.tensor(img.reshape((1,*img.shape)))
    
    
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
     
    # preprocessing, making dataset and loader
    
    
    
    #test_loader = DataLoader(dataset=test_dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=8)
    
    # computing metrics
    score, prediction = evaluate(model, img_tensor, config, device)
     
    print(f"Score: {score}")
    print(f"Prediction class: {prediction} - {'living' if prediction[0] else 'spoof'}")
    
    print ("Done!")
    
def evaluate(model, img, config, device):
    ''' evaluating AUC, EER, BPCER, APCER, ACER on given data loader and model '''
    model.eval()
    
    image = img.to(device)
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
    return positive_probabilities, y_pred
    
if __name__ == "__main__":
    main()

