import argparse
import json
import os
import pickle
from pathlib import Path

from tqdm import tqdm


TARTAN_LEFT_CAM_IN = [
    477.6049499511719,
    477.6049499511719,
    499.5,
    252.0,
]


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def depth_path_for_image(out_root, tartan_root, image_path):
    image_path = Path(image_path).resolve()
    rel = image_path.relative_to((tartan_root / 'data').resolve())
    return out_root / rel.with_suffix('.npy')


def build_annotations(root, splits, out_root):
    files = []
    seen = set()
    for split in splits:
        meta_path = root / f'meta_{split}' / f'{split}_meta.json'
        for item in tqdm(load_json(meta_path), desc=f'meta_{split}'):
            data_root = item['data_root']
            cam_dir = item.get('cam_dir', 'image_left_color')
            rel_folder = f'{data_root}/{cam_dir}'
            for filename in item.get('CAM_F0', []):
                image_path = root / 'data' / data_root / cam_dir / filename
                if not image_path.exists():
                    continue
                image_key = str(image_path.resolve())
                if image_key in seen:
                    continue
                seen.add(image_key)
                rel_filename = f'{rel_folder}/{filename}'
                files.append(
                    dict(
                        rgb=image_key,
                        filename=rel_filename,
                        folder=rel_folder,
                        cam_in=TARTAN_LEFT_CAM_IN,
                        depth_scale=256.0,
                        depth_output=str(depth_path_for_image(out_root, root, image_path)),
                    )
                )
    return files


def update_info_pkls(info_root, source_root, depth_root, splits):
    for split in splits:
        pkl_path = info_root / f'vad_tartan_drive_infos_temporal_{split}.pkl'
        if not pkl_path.exists():
            continue
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        update_count = 0
        for info in data.get('infos', []):
            for cam_info in info.get('cams', {}).values():
                image_path = cam_info.get('data_path')
                if not image_path:
                    continue
                cam_info['depth_path'] = str(depth_path_for_image(depth_root, source_root, image_path))
                update_count += 1
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'updated {update_count} camera depth paths in {pkl_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Create Metric3D annotations for TartanDrive.')
    parser.add_argument('--root', type=Path, default=Path('data_world_model/tartan_drive_2.0'))
    parser.add_argument('--splits', nargs='+', default=['train', 'test'], choices=['train', 'test'])
    parser.add_argument('--out', type=Path, default=Path('tartan_drive.json'))
    parser.add_argument('--depth-root', type=Path, default=Path('data/tartan_drive_2.0/depth_gaussian'))
    parser.add_argument('--update-info-pkls', action='store_true')
    parser.add_argument('--info-root', type=Path, default=Path('data/tartan_drive_2.0'))
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    depth_root = args.depth_root.resolve()
    files = build_annotations(root, args.splits, depth_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(dict(files=files), f)
    print(f'wrote {len(files)} samples to {args.out}')
    print(f'depths will be saved under {depth_root}')
    if args.update_info_pkls:
        update_info_pkls(args.info_root.resolve(), root, depth_root, args.splits)


if __name__ == '__main__':
    main()
