exp_num = 0

dataset = 'replay_attack'

multi_task_learning = False

evaluation = True

test_steps = None

#txt sources
dataset_files = dict(root_folder="./datasets/replay-attack_training",
                     train_data_folder="Caffe_Data_train_Replay-attack",
                     train_txt_filename="train_replay_attack.txt",
                     test_data_folder="Caffe_Data_val_Replay-attack",
                     test_txt_filename="val_replay_attack.txt",
                     val_data_folder="Caffe_Data_val_Replay-attack",
                     val_txt_filename="val_replay_attack.txt")

datasets = dict(LCCFASD_root='./LCC_FASDcropped',
                Celeba_root='/home/lusantlueg/Documents/light-weight-face-anti-spoofing/datasets/CelebA_Spoof/',
                Casia_root='./CASIA',
                replay_attack=dataset_files)

external = dict(train=dict(), val=dict(), test=dict())

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
                    std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,40], gamma=0.2)

data = dict(batch_size=128,
            data_loader_workers=8,
            sampler=None,
            pin_memory=True)

resize = dict(height=128, width=128)

checkpoint = dict(snapshot_name="MobileNet2.pth.tar",
                  experiment_path='./logs')

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

model= dict(model_type='Mobilenet3',
            model_size = 'large',
            width_mult = 1.0,
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

test_dataset = dict(type='replay_attack')

conv_cd = dict(theta=0)
