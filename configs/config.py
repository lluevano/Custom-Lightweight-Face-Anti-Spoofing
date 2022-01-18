exp_num = 0

dataset = 'celeba_spoof'

multi_task_learning = True

evaluation = True

test_steps = None

datasets = dict(LCCFASD_root='./LCC_FASDcropped',
                Celeba_root='/media/lusantlueg/M.2 Fast Disk/light-weight-face-anti-spoofing/datasets/CelebA_Spoof/',
                Casia_root='./CASIA')

external = dict(train=dict(), val=dict(), test=dict())

img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
                    std=[0.2471, 0.2214, 0.2157])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[20,40], gamma=0.2)

data = dict(batch_size=512,
            data_loader_workers=16,
            sampler=None,
            pin_memory=True)

resize = dict(height=128, width=128)

checkpoint = dict(snapshot_name="MobileNet2_1.0.pretrained_imagenet_128x128.pth.tar",
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

model= dict(model_type='Mobilenet2',
            model_size = 'large',
            width_mult = 1.0,
            pretrained=True,
            embeding_dim=1280,
            imagenet_weights='./models/mobilenetv2_128x128-fd66a69d.pth'
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

data_parallel = dict(use_parallel=True,
                     parallel_params=dict(device_ids=[0,1], output_device=0))

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='celeba_spoof')

conv_cd = dict(theta=0)
