
from skimage import transform as trans
import cv2
import numpy as np
np.bool = bool
import glob2
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from multiprocessing import cpu_count, Pool, Manager, Lock
from functools import partial

import mxnet as mx
from face_detection import FaceDetector

class Pickable: 
    def __init__(self):
        self.detector = FaceDetector('../../models/retinaface-R50/R50-0000.params', rac='net3')
        self.detector.prepare(ctx_id=-1,nms=0.2)

def detect_faces(img):
    #sym, arg_params, aux_params= mx.model.load_checkpoint('/root/.insightface/models/retinaface-R50/R50', 0)

    
    bbox, landmarks = detector.detect(img)
    return bbox, landmarks

#  Code taken from https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py
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

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
    
#  Stop code from insightface

def prepare_train_set(detector, img_path):
    #living_metadata_iter = glob2.iglob("UniAttackData/phase1/p1/train")
    #spoof_metadata_iter = glob2.iglob("train/spoof/*/*/*.txt")
    splt_path = img_path.split('/')
    img_name = splt_path[-1]
    norm_crop_dir = os.path.join('norm_crop', *(splt_path[:-1]))
    img_norm_crop_path = os.path.join(norm_crop_dir, img_name)
    if os.path.exists(img_norm_crop_path):
        return

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not open {img_path}")
        exit(-1)
    
    img_rgb = np.ascontiguousarray(img[..., ::-1])
    bbox, landmarks = detector.detect(img_rgb)

    n_faces = np.shape(bbox)[0]

    if not n_faces:
        print(f"Could not find face for {img_path}. Resizing and saving as is")
        img_norm_crop = cv2.resize(img_rgb, (128,128), interpolation = cv2.INTER_AREA)
    else:
        img_norm_crop = norm_crop(img_rgb, landmarks[0], image_size=128)
    #top_left = [bbox[0][0], bbox[0][1]] # (X,Y)
    #bottom_right = [bbox[0][2], bbox[0][3]] # (X,Y)

    #top_left[0] = max(0, top_left[0])
    #top_left[1] = max(0, top_left[1])
    #bottom_right[0] = min(bottom_right[0], img_rgb.shape[1])
    #bottom_right[1] = min(bottom_right[1], img_rgb.shape[0])

    #img_crop = img_rgb[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0],:]
    


    
    #lock.acquire()
    os.makedirs(norm_crop_dir, exist_ok=True)
    if not cv2.imwrite(img_norm_crop_path, cv2.cvtColor(img_norm_crop, cv2.COLOR_RGB2BGR)):
        print(f"Cannot save {img_norm_crop_path}")
        return
    #lock.release()
    #file_list+=img_norm_crop_path+"\n"
    #iters_label = {0: spoof_metadata_iter, 1: living_metadata_iter}

    #with open(os.path.join(".","norm_crop.txt"),'w') as f_norm_crop:
    #    f_norm_crop.write(file_list.strip())


    
    #print("Done")


#def init(l):
#    global lock
#    lock = l

if __name__ == '__main__':
    #lst_file = glob2.glob(os.path.join('**','*.dds'))

    

    lst_imgs = glob2.glob("UniAttackData/*/*/*/*")
    #lock = Lock()
    #pool = Pool(cpu_count() - 1, initializer=init, initargs=(lock,))
    #task = partial(prepare_train_set, Pickable())
    detector = Pickable().detector
    for e in lst_imgs:
        prepare_train_set(detector, e)
    #pool.map(task, lst_imgs)
    #pool.join()
    #pool.close()
    print("Done!")
