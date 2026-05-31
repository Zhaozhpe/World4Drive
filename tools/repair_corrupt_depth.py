#!/usr/bin/env python
"""Find and regenerate corrupt Metric3D depth npy files.

This script scans expected NuScenes camera depth outputs referenced by W4D info
pkls. Bad files are detected by actually loading the npy, because truncated npy
files can have a valid header but fail during payload reshape.

Usage:

# Just check corrupte files without modifying anything
python tools/repair_corrupt_depth.py --dry-run

# Run full repair by default
python tools/repair_corrupt_depth.py --gpu 7

# Only scan/repair val:
python tools/repair_corrupt_depth.py \
  --ann-files data/nuscenes/vad_nuscenes_infos_temporal_val.pkl \
  --gpu 7
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np


CAMERAS = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)


def load_infos(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['infos']


def depth_path_for_image(depth_root, image_path):
    return depth_root / f'{Path(image_path).stem}.npy'


def check_depth(path, expected_shape):
    if not path.exists():
        return 'missing'
    try:
        depth = np.load(path)
        if expected_shape is not None and tuple(depth.shape[:2]) != expected_shape:
            return f'bad_shape:{tuple(depth.shape)}'
        if not np.isfinite(depth).all():
            return 'non_finite'
    except Exception as exc:
        return repr(exc)
    return None


def collect_bad_depths(args):
    depth_root = Path(args.depth_root)
    expected_shape = None if args.expected_shape is None else tuple(args.expected_shape)
    records = {}
    checked = 0

    for ann_file in args.ann_files:
        for info_idx, info in enumerate(load_infos(Path(ann_file))):
            print(f'[scan] info_idx={info_idx}/{len(load_infos(Path(ann_file)))}', end='\r')
            for cam in CAMERAS:
                cam_info = info['cams'][cam]
                image_path = cam_info['data_path']
                depth_path = depth_path_for_image(depth_root, image_path)
                reason = check_depth(depth_path, expected_shape)
                checked += 1
                if reason is None:
                    continue
                key = str(depth_path)
                records[key] = dict(
                    depth_output=key,
                    rgb=image_path,
                    cam_in=[
                        float(cam_info['cam_intrinsic'][0, 0]),
                        float(cam_info['cam_intrinsic'][1, 1]),
                        float(cam_info['cam_intrinsic'][0, 2]),
                        float(cam_info['cam_intrinsic'][1, 2]),
                    ],
                    depth_scale=256.0,
                    ann_file=str(ann_file),
                    info_idx=info_idx,
                    token=info.get('token'),
                    cam=cam,
                    reason=reason,
                )
                if args.limit and len(records) >= args.limit:
                    return checked, list(records.values())

    return checked, list(records.values())


def write_json(path, records):
    payload = {
        'files': [
            dict(
                rgb=record['rgb'],
                cam_in=record['cam_in'],
                depth_scale=record['depth_scale'],
                depth_output=record['depth_output'],
            )
            for record in records
        ]
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def backup_bad_outputs(records):
    for record in records:
        path = Path(record['depth_output'])
        if not path.exists():
            continue
        backup = path.with_suffix(path.suffix + f'.corrupt.{int(time.time())}')
        path.rename(backup)
        record['backup'] = str(backup)


def run_regeneration(args, manifest_path):
    cmd = [
        sys.executable,
        'mono/tools/test_scale_cano.py',
        args.config,
        '--load-from',
        args.checkpoint,
        '--test_data_path',
        str(manifest_path),
        '--launcher',
        'None',
        '--batch_size',
        str(args.batch_size),
        '--begin',
        '0',
        '--end',
        str(args.end if args.end is not None else 10**9),
    ]
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-files',
        nargs='+',
        default=[
            'data/nuscenes/vad_nuscenes_infos_temporal_train.pkl',
            'data/nuscenes/vad_nuscenes_infos_temporal_val.pkl',
        ])
    parser.add_argument('--depth-root', default='data/depth_gaussian')
    parser.add_argument('--expected-shape', type=int, nargs=2, default=(450, 800),
                        metavar=('H', 'W'))
    parser.add_argument('--manifest', default='work_dirs/corrupt_depth_repair.json')
    parser.add_argument('--report', default='work_dirs/corrupt_depth_report.json')
    parser.add_argument('--config', default='mono/configs/HourglassDecoder/vit.raft5.giant2.py')
    parser.add_argument('--checkpoint', default='weight/metric_depth_vit_giant2_800k.pth')
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0,
                        help='Scan stops after this many bad files. 0 means no limit.')
    parser.add_argument('--end', type=int, default=None,
                        help='Optional end value passed to Metric3D. Defaults to all manifest entries.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only write report/manifest; do not move files or run Metric3D.')
    args = parser.parse_args()

    checked, records = collect_bad_depths(args)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(dict(checked=checked, bad_count=len(records), bad=records), f, indent=2)
    write_json(args.manifest, records)

    print(f'checked={checked} bad={len(records)}')
    print(f'wrote report: {args.report}')
    print(f'wrote Metric3D manifest: {args.manifest}')

    if not records:
        return
    if args.dry_run:
        print('dry-run: not moving corrupt files or regenerating')
        return

    backup_bad_outputs(records)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(dict(checked=checked, bad_count=len(records), bad=records), f, indent=2)
    run_regeneration(args, args.manifest)


if __name__ == '__main__':
    main()
