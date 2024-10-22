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

from functools import partial

from .celeba_spoof import CelebASpoofDataset
from .casia_surf import CasiaSurfDataset
from .text_folder_dataset import TextFolderDataset
from .lcc_fasd import LccFasdDataset
from .uniattackdata import UniAttackData

def do_nothing(**args):
    pass

def read_from_txt(**args):
    pass

# import your reader and replace do_nothing with it
external_reader=do_nothing

def get_datasets(config):

    celeba_root = config.datasets.Celeba_root
    lccfasd_root = config.datasets.LCCFASD_root
    casia_root = config.datasets.Casia_root

    #set of datasets
    datasets = {'celeba_spoof_train': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=False,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_val': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=True,
                                            multi_learning=config.multi_task_learning),

                'celeba_spoof_test': partial(CelebASpoofDataset, root_folder=celeba_root,
                                            test_mode=True, multi_learning=config.multi_task_learning),

                'Casia_train': partial(CasiaSurfDataset, protocol=1, dir_=casia_root,
                                    mode='train'),

                'Casia_val': partial(CasiaSurfDataset, protocol=1, dir_=casia_root,
                                    mode='dev'),

                'Casia_test': partial(CasiaSurfDataset, protocol=1, dir_=casia_root, mode='test'),

                #'LCC_FASD_train': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='train'),

#                'LCC_FASD_val': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val'),

 #               'LCC_FASD_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='test'),

#                'LCC_FASD_val_test': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='val_test'),

 #               'LCC_FASD_combined': partial(LccFasdDataset, root_dir=lccfasd_root, protocol='combine_all'),

                #'external_train': partial(external_reader, **config.external.train_params),

                #'external_val': partial(external_reader, **config.external.val_params),

                #'external_test': partial(external_reader, **config.external.test_params)
                'replay_attack_train': partial(TextFolderDataset, root_folder=config.datasets.replay_attack.root_folder,
                                                data_folder=config.datasets.replay_attack.train_data_folder,
                                                txt_filename=config.datasets.replay_attack.train_txt_filename),
                'replay_attack_test': partial(TextFolderDataset, root_folder=config.datasets.replay_attack.root_folder,
                                               data_folder=config.datasets.replay_attack.test_data_folder,
                                               txt_filename=config.datasets.replay_attack.test_txt_filename),
                'replay_attack_val': partial(TextFolderDataset, root_folder=config.datasets.replay_attack.root_folder,
                                              data_folder=config.datasets.replay_attack.val_data_folder,
                                              txt_filename=config.datasets.replay_attack.val_txt_filename),
                'FASCVPR2023_train': partial(TextFolderDataset, root_folder=config.datasets.FASCVPR2023.root_folder,
                                                data_folder=config.datasets.FASCVPR2023.train_data_folder,
                                                txt_filename=config.datasets.FASCVPR2023.train_txt_filename, multi_learning=config.multi_spoof),
                'FASCVPR2023_val': partial(TextFolderDataset, root_folder=config.datasets.FASCVPR2023.root_folder,
                                              data_folder=config.datasets.FASCVPR2023.val_data_folder,
                                              txt_filename=config.datasets.FASCVPR2023.val_txt_filename),
                'FASCVPR2023_test': partial(TextFolderDataset, root_folder=config.datasets.FASCVPR2023.root_folder,
                                                data_folder=config.datasets.FASCVPR2023.test_data_folder,
                                                txt_filename=config.datasets.FASCVPR2023.test_txt_filename),
                'CelebA_spoof_norm_crop_train': partial(TextFolderDataset, root_folder=config.datasets.CelebA_spoof_norm_crop.root_folder,
                                                data_folder=config.datasets.CelebA_spoof_norm_crop.train_data_folder,
                                                txt_filename=config.datasets.CelebA_spoof_norm_crop.train_txt_filename),
                'CelebA_spoof_norm_crop_val': partial(TextFolderDataset, root_folder=config.datasets.FASCVPR2023.root_folder,
                                                data_folder=config.datasets.FASCVPR2023.val_data_folder,
                                                txt_filename=config.datasets.FASCVPR2023.val_txt_filename),
                'CelebA_spoof_norm_crop_test': partial(TextFolderDataset, root_folder=config.datasets.CelebA_spoof_norm_crop.root_folder,
                                                data_folder=config.datasets.CelebA_spoof_norm_crop.test_data_folder,
                                                txt_filename=config.datasets.CelebA_spoof_norm_crop.test_txt_filename),
                'OULU_NPU_4_train': partial(TextFolderDataset, root_folder=config.datasets.OULU_NPU_4.root_folder,
                                                data_folder=config.datasets.OULU_NPU_4.train_data_folder,
                                                txt_filename=config.datasets.OULU_NPU_4.train_txt_filename, multi_learning=config.multi_spoof),
                'OULU_NPU_4_val': partial(TextFolderDataset, root_folder=config.datasets.OULU_NPU_4.root_folder,
                                              data_folder=config.datasets.OULU_NPU_4.val_data_folder,
                                              txt_filename=config.datasets.OULU_NPU_4.val_txt_filename),
                'OULU_NPU_4_test': partial(TextFolderDataset, root_folder=config.datasets.OULU_NPU_4.root_folder,
                                                data_folder=config.datasets.OULU_NPU_4.test_data_folder,
                                                txt_filename=config.datasets.OULU_NPU_4.test_txt_filename),
                'UniAttackData_train': partial(UniAttackData, root_folder=config.datasets.UniAttackData.root_folder,
                                                protocol=config.datasets.UniAttackData.protocol,
                                                subset="train",
                                                phase="phase1"),
                'UniAttackData_val': partial(UniAttackData, root_folder=config.datasets.UniAttackData.root_folder,
                                                protocol=config.datasets.UniAttackData.protocol,
                                                subset="dev",
                                                phase="phase1"),
                'UniAttackData_test': partial(UniAttackData, root_folder=config.datasets.UniAttackData.root_folder,
                                                protocol=config.datasets.UniAttackData.protocol,
                                                subset="dev",
                                                phase="phase1"),
                }
    return datasets
