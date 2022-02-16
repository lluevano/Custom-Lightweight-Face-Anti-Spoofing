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

import json
import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd



class TextFolderDataset(Dataset):
    def __init__(self, root_folder, data_folder, txt_filename=False, transform=None, multi_learning=False):
        self.root_folder = root_folder
        self.data_folder = data_folder

        list_path = os.path.join(root_folder, txt_filename)
        self.data = pd.read_csv(list_path, sep=" ", header=None, names=["filename", "class"])
        #print(root_folder)
        #print(txt_filename)
        #print(self.data)
        # transform is supposed to be instance of the Transform object from utils.py pending entry label
        self.transform = transform
        self.multi_learning = multi_learning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            data_path = self.data["filename"][idx]
            img = cv.imread(os.path.join(self.root_folder, self.data_folder, data_path))
            #print("Successfully loaded idx ",idx)
        except:
            print("Skipping index ", idx)
            return None

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.multi_learning:
            raise "Multi learning not implemented error"
        else:
            label = int(self.data["class"][idx])

        if self.transform:
            img = self.transform(label=label, img=img)['image']
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return (torch.tensor(img), torch.tensor(label, dtype=torch.long))


