exp_num = 0

dataset = 'UniAttackData'

multi_task_learning = False

multi_spoof = False
criterion_params = dict(C=1.0, Cs=0.25)

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
                     train_txt_filename="train_norm_crop.txt", #prev was train_norm_crop.txt , split_train.txt
                     test_data_folder="",
                     test_txt_filename="split_val.txt",
                     val_data_folder="",
                     val_txt_filename="split_val.txt")

CelebA_spoof_files = dict(root_folder="./datasets/CelebA_spoof",
                     train_data_folder="",
                     train_txt_filename="train_norm_crop.txt", #prev was train_norm_crop.txt , split_train.txt
                     test_data_folder="",
                     test_txt_filename="test_norm_crop.txt",
                     #val_data_folder="",
                     #val_txt_filename="split_val.txt
                     )

OULU_NPU_4_files = dict(root_folder="./datasets/OULU_NPU/Protocol_4",
                     train_data_folder="",
                     train_txt_filename="norm_crop_5_Train.txt",
                     test_data_folder="",
                     test_txt_filename="norm_crop_5_Test.txt",
                     val_data_folder="",
                     val_txt_filename="norm_crop_5_Dev.txt")

UniAttackData_files = dict(root_folder="./datasets/FAS-CVPR2024/norm_crop/UniAttackData/",
                     protocol="p1")



datasets = dict(LCCFASD_root='./LCC_FASDcropped',
                Celeba_root='/home/lusantlueg/Documents/light-weight-face-anti-spoofing/datasets/CelebA_Spoof/',
                Casia_root='./CASIA',
                replay_attack=replay_attack_files,
                FASCVPR2023=FASCVPR2023_files,
                CelebA_spoof_norm_crop=CelebA_spoof_files,
                OULU_NPU_4=OULU_NPU_4_files,
                UniAttackData=UniAttackData_files)

external = dict(train=dict(), val=dict(), test=dict())

#img_norm_cfg = dict(mean=[0.5931, 0.4690, 0.4229],
#                    std=[0.2471, 0.2214, 0.2157]) #replay attack
                    
#img_norm_cfg = dict(mean=[126.4611/255.0, 107.1148/255.0, 100.2191/255.0], 
#                    std=[68.6844/255.0, 62.5515/255.0, 61.6850/255.0]) #FASCVPR2023  

img_norm_cfg = dict(mean=[65.6124/255.0, 49.0762/255.0, 44.2558/255.0], 
                    std=[62.2802/255.0, 50.2993/255.0, 48.7784/255.0]) #FASCVPR2024_train_p1  

#img_norm_cfg = dict(mean=[63.2480/255.0, 47.6288/255.0, 43.1594/255.0], 
#                    std=[62.7932/255.0, 50.6263/255.0, 49.0550/255.0]) #FASCVPR2024_train_p2.1

#img_norm_cfg = dict(mean=[58.2901/255.0, 40.0298/255.0, 34.6466/255.0], 
#                    std=[67.2511/255.0, 49.0045/255.0, 44.7642/255.0]) #FASCVPR2024_train_p2.2

#img_norm_cfg = dict(mean=[129.2820/255.0, 107.9376/255.0, 99.6690/255.0], std=[72.0941/255.0, 65.0965/255.0, 63.7481/255.0]) #celebaspoof   

optimizer = dict(lr=0.0001, momentum=0.9, weight_decay=5e-4)

scheduler = dict(milestones=[200,400], gamma=0.2)

data = dict(batch_size=1024,
            data_loader_workers=16,
            sampler=None,
            pin_memory=True)

resize = dict(height=224, width=224)


loss = dict(loss_type='amsoftmax',
            amsoftmax=dict(m=0.5,
                           s=1,
                           margin_type='cross_entropy',
                           label_smooth=False,
                           smoothing=0.1,
                           ratio=[1,1],
                           gamma=0),
            soft_triple=dict(cN=2, K=10, s=1, tau=.2, m=0.35))

epochs = dict(start_epoch=0, max_epoch=500)

activation="relu"

model= dict(model_type='Mobilenet3',
            model_size = 'large',
            width_mult = 1.25,
            pretrained=False,
            embeding_dim=1280,
            imagenet_weights=None
            )
            
checkpoint = dict(snapshot_name=f"{model['model_type']}_{model['model_size']}_{activation}_{dataset}_p1.pth.tar",
                  experiment_path='./logs')

aug = dict(type_aug=None,
            alpha=0.5,
            beta=0.5,
            aug_prob=0.7)

curves = dict(det_curve='det_curve_0.png',
              roc_curve='roc_curve_0.png')

dropout = dict(prob_dropout=0.1, #ignoring on micronet
               classifier=0.35,
               type='bernoulli',
               mu=0.5,
               sigma=0.3)

data_parallel = dict(use_parallel=False,
                     parallel_params=dict(device_ids=[0], output_device=0))

RSC = dict(use_rsc=False,
           p=0.333,
           b=0.333)

test_dataset = dict(type='UniAttackData')

conv_cd = dict(theta=0)
