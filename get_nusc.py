import pickle
from tqdm import tqdm
import json
import os
# 读取pickle
# pickle_path = "/home/csgrad/zzhao43/world_model/World4Drive/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl"
pickle_path = "/home/csgrad/zzhao43/world_model/World4Drive/data/nuscenes/vad_nuscenes_infos_temporal_train.pkl"
with open(pickle_path, 'rb') as f:
    infos = pickle.load(f)

file_list = []

for info in tqdm(infos["infos"]):
    for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        sample_dict = {}
        cam_path = info['cams'][cam]['data_path'] #.replace("/data42/DATASET/nuscenes", "/data11/zyp/Driving/TOD3Cap/tod3cap_camera/data/nuscenes")
        # print("cam_path", cam_path)
        assert cam_path == info['cams'][cam]['data_path'], "replacement happened"
        assert os.path.exists(cam_path), f"cam_path does not exist: {cam_path}"
        intrinsic = info['cams'][cam]['cam_intrinsic']
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        cam_in = [fx, fy, cx, cy]
        sample_dict['cam_in'] = cam_in
        sample_dict['rgb'] = cam_path
        # sample_dict['depth'] = cam_path
        sample_dict['depth_scale'] = 256.0
        file_list.append(sample_dict)
save_dict = {}
save_dict['files'] = file_list
print(f"Total samples: {len(file_list)}")
# 存json
# with open('nuscenes_val.json', 'w') as f:
#     json.dump(save_dict, f)

with open('nuscenes_train.json', 'w') as f:
    json.dump(save_dict, f)