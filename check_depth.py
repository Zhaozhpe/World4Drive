import os
import numpy as np
import pickle
from tqdm import tqdm

pickle_path = "/data11/zyp/Driving/TOD3Cap/tod3cap_camera/data/nuscenes/nuscenes_infos_train.pkl"
with open(pickle_path, 'rb') as f:
    infos = pickle.load(f)
    
for info in tqdm(infos["infos"]):
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        cam_path = info['cams'][cam]['data_path'].replace(f"/data42/DATASET/nuscenes/samples/{cam}", "depth")
        cam_path = cam_path.replace(".jpg", ".npy")
        try:
            d = np.load(cam_path)
        except:
            print(cam_path)