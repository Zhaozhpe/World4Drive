import glob
import os
import json
import cv2
import numpy as np
def load_from_annos(anno_path):
    with open(anno_path, 'r') as f:
        annos = json.load(f)['files']

    datas = []
    for i, anno in enumerate(annos):
        rgb = anno['rgb'].replace("/data11/zyp/Driving/TOD3Cap/tod3cap_camera/data/nuscenes/samples", "/data41/zyp/Driving/exp_rebuttal/gaussian_blur_samples")
        if os.path.exists(rgb):
            depth = anno['depth'] if 'depth' in anno else None
            depth_scale = anno['depth_scale'] if 'depth_scale' in anno else 1.0
            intrinsic = anno['cam_in'] if 'cam_in' in anno else None
            normal = anno['normal'] if 'normal' in anno else None

            filename = anno['filename'] if 'filename' in anno else os.path.basename(rgb)
            folder = anno['folder'] if 'folder' in anno else rgb.split('/')[-2]
            data_i = {
                'rgb': rgb,
                'depth': depth,
                'depth_output': anno.get('depth_output'),
                'depth_scale': depth_scale,
                'intrinsic': intrinsic,
                'filename': filename,
                'folder': folder,
                'normal': normal
            }
            datas.append(data_i)
    return datas

def load_data(path: str):
    rgbs = glob.glob(path + '/*.jpg') + glob.glob(path + '/*.png')
    data = [{'rgb': i, 'depth': None, 'intrinsic': None, 'filename': os.path.basename(i), 'folder': i.split('/')[-3]} for i in rgbs]
    return data