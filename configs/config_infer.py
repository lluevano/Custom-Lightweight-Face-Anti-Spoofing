exp_num = 0

dataset = 'FASCVPR2023'

multi_task_learning = False

evaluation = True

test_steps = None

#txt sources
replay_attack_files = dict(root_folder="./datasets/replay-attack_training",
                     train_data_folder="Caffe_Data_train_Replay-attack",
                     train_txt_filename="train_replay_attack.txt",
                     test_data_folder="Caffe_Data_val_Replay-attack",
                     test_txt_filename="val_replay_attack_cvpr2023.txt",
                     val_data_folder="Caffe_Data_val_Replay-attack",
                     val_txt_filename="val_replay_attack_cvpr2023.txt")
                     
FASCVPR2023_files = dict(root_folder="./datasets/FAS-CVPR2023",
                     train_data_folder="",
                     train_txt_filename="train_norm_crop.txt")

datasets = dict(LCCFASD_root='./LCC_FASDcropped',
                Celeba_root='/home/lusantlueg/Documents/light-weight-face-anti-spoofing/datasets/CelebA_Spoof/',
                Casia_root='./CASIA',
                replay_attack=replay_attack_files,
                FASCVPR2023=FASCVPR2023_files)

external = dict(train=dict(), val=dict(), test=dict())

#img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
#                    std=[0.2471, 0.2214, 0.2157]) #replay attack
                    
img_norm_cfg = dict(mean=[126.4611/255.0, 107.1148/255.0, 100.2191/255.0], 
                    std=[68.6844/255.0, 62.5515/255.0, 61.6850/255.0]) #FASCVPR2023  

activation = "dyshiftmax"

optimizer = dict(lr=0.1, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,40], gamma=0.2)

data = dict(batch_size=1024,
            data_loader_workers=16,
            sampler=None,
            pin_memory=True)

resize = dict(height=224, width=224)

checkpoint = dict(snapshot_name="best_65_Micronet_M3_CVPR2023.pth.tar",
                  experiment_path='./logs/split_train/micronet-M3')

loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5,
                           s=1,
                           margin_type='cross_entropy',
                           label_smooth=False,
                           smoothing=0.1,
                           ratio=[1,1],
                           gamma=0),
            soft_triple=dict(cN=2, K=10, s=1, tau=.2, m=0.35))

epochs = dict(start_epoch=0, max_epoch=71)

model= dict(model_type='Micronet',
            model_size = 'M3',
            width_mult = None,
            pretrained=False,
            embeding_dim=1280,
            imagenet_weights=None
            )

aug = dict(type_aug=None,
            alpha=0.5,
            beta=0.5,
            aug_prob=0.7)

curves = dict(det_curve='det_curve_0.png',
              roc_curve='roc_curve_0.png')

dropout = dict(prob_dropout=0.1,
               classifier=0.35,
               type='bernoulli',
               mu=0.5,
               sigma=0.3)

data_parallel = dict(use_parallel=False,
                     parallel_params=dict(device_ids=[0,1], output_device=0))

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='FAS2023_DEV')

conv_cd = dict(theta=0)
