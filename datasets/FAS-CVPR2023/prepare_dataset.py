import glob2
import cv2
import os
import numpy as np
from face_align import norm_crop

def prepare_dev_set():
    metadata_iter = glob2.iglob("./dev/data/*.txt")

    n_imgs = 0

    dev_list = ""
    crop_list = ""
    norm_crop_list = ""

    for metadata in metadata_iter:
        with open(metadata, 'r') as f:
            
            content = f.readlines()
            
            
            top_left = [int(pos) for pos in content[0].strip().split(" ")] # (X,Y)
            bottom_right = [int(pos) for pos in content[1].strip().split(" ")] # (X,Y)
            
            landmarks = []
            for i in range(2, 7):
                landmarks.append([float(pos) for pos in content[i].strip().split(" ")])
            
            landmarks = np.array(landmarks, dtype=np.float32)
            
            img_path = metadata[:-4]+".jpg"
            img_name = img_path.split('/')[-1]  
            
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            #print(f"loading image {img_path} with dimensions {str(img.shape)}")
            
            #clip bounding boxes
            top_left[0] = max(0, top_left[0])
            top_left[1] = max(0, top_left[1])
            bottom_right[0] = min(bottom_right[0], img.shape[1])
            bottom_right[1] = min(bottom_right[1], img.shape[0])
            
            img_crop = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
            
            if not cv2.imwrite(os.path.join(".","dev","cropped_data",img_name),img_crop):
                print(f"Cannot save {img_name} in cropped_data")
                exit(-1)
            
            
            
            img_norm_crop = norm_crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks, image_size=128)
            
            if not cv2.imwrite(os.path.join(".","dev","norm_crop",img_name),cv2.cvtColor(img_norm_crop, cv2.COLOR_RGB2BGR)):
                print(f"Cannot save {img_name} in norm_crop")
                exit(-1)
             
            dev_list += (os.path.join("dev","data",img_name)+'\n')
            crop_list += (os.path.join("dev","cropped_data",img_name)+'\n')
            norm_crop_list += (os.path.join("dev","norm_crop",img_name)+'\n')
            
    with open(os.path.join(".","dev","dev_data.txt"),'w') as f_dev_data:
                f_dev_data.write(dev_list.strip())
    
    with open(os.path.join(".","dev","cropped_data.txt"),'w') as f_cropped_data:
                f_cropped_data.write(crop_list.strip())
                
    with open(os.path.join(".","dev","norm_crop.txt"),'w') as f_norm_crop:
                f_norm_crop.write(norm_crop_list.strip())
                
                
    print("Done")
    
def prepare_train_set():
    living_metadata_iter = glob2.iglob("train/living/*/*.txt")
    spoof_metadata_iter = glob2.iglob("train/spoof/*/*/*.txt")
    
    n_imgs = 0

    iters_label = {0: spoof_metadata_iter, 1: living_metadata_iter}

    
    data_list = ""
    crop_list = ""
    norm_crop_list = ""

    print ("Start iteration")

    for key in iters_label.keys():
        
        for metadata in iters_label[key]:
            with open(metadata, 'r') as f:
                content = f.readlines()
                
                
                top_left = [int(pos) for pos in content[0].strip().split(" ")] # (X,Y)
                bottom_right = [int(pos) for pos in content[1].strip().split(" ")] # (X,Y)
                
                landmarks = []
                for i in range(2, 7):
                    landmarks.append([float(pos) for pos in content[i].strip().split(" ")])
                
                landmarks = np.array(landmarks, dtype=np.float32)
                
                img_path = metadata[:-4]+".jpg"
                img_name = img_path.split('/')[-1]  
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                    
                crop_dir = os.path.join("train","crop",*(img_path.split('/')[1:-1]))
                img_crop_path = os.path.join(crop_dir, img_name)
                
                norm_crop_dir = os.path.join("train","norm_crop",*(img_path.split('/')[1:-1]))
                img_norm_crop_path = os.path.join(norm_crop_dir, img_name)
                
                data_list += (img_path+" "+str(key)+'\n')
                crop_list += (img_crop_path+" "+str(key)+'\n')
                norm_crop_list += (img_norm_crop_path+" "+str(key)+'\n')
                
                if (os.path.exists(img_crop_path) and os.path.exists(img_norm_crop_path)):
                    continue
                
                #print(f"loading image {img_path} with dimensions {str(img.shape)}")
                
                #clip bounding boxes
                top_left[0] = max(0, top_left[0])
                top_left[1] = max(0, top_left[1])
                bottom_right[0] = min(bottom_right[0], img.shape[1])
                bottom_right[1] = min(bottom_right[1], img.shape[0])
                
                img_crop = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
                
                
                os.makedirs(crop_dir, exist_ok=True)
                if not cv2.imwrite(img_crop_path, img_crop):
                    print(f"Cannot save {img_crop_path}")
                    exit(-1)
                
                img_norm_crop = norm_crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), landmarks, image_size=128)
                
                
                os.makedirs(norm_crop_dir, exist_ok=True)
                if not cv2.imwrite(img_norm_crop_path, cv2.cvtColor(img_norm_crop, cv2.COLOR_RGB2BGR)):
                    print(f"Cannot save {img_norm_crop_path}")
                    exit(-1)
                
        print(f"Finished key {key}")
    with open(os.path.join(".","train","train_data.txt"),'w') as f_train_data:
                f_train_data.write(data_list.strip())
    
    with open(os.path.join(".","train","crop_data.txt"),'w') as f_crop_data:
                f_crop_data.write(crop_list.strip())
                
    with open(os.path.join(".","train","norm_crop.txt"),'w') as f_norm_crop:
                f_norm_crop.write(norm_crop_list.strip())
                
                
    print("Done")
    
    
    
prepare_train_set()
    
    
