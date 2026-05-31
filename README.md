# 🚀 Metric3D Project 🚀

**Generate Metric Depth for nuScenes with Metric3Dv1 and Metric3Dv2:**   

## Run
```
# generate data json for metric 3d inference
python get_nusc.py
```

```
# generate depth map, ckpts:
# weight/metric_depth_vit_giant2_800k.pth
# weight/metric_depth_vit_small_800k.pth
# weight/metric_depth_vit_large_800k.pth

sh gen.sh

# if you find any corrupt files, please refer to ./tools/repair_corrupt_depth.py to scan and repair them
```

## 📧 Citation
Please cite Metric 3D papers if this help your research.
```
@article{hu2024metric3dv2,
  title={Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation},
  author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
```
@article{yin2023metric,
  title={Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image},
  author={Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen},
  booktitle={ICCV},
  year={2023}
}
```

## License and Contact

The *Metric 3D* code is under a 2-clause BSD License. For further commercial inquiries, please contact Dr. Wei Yin  [yvanwy@outlook.com] and Mr. Mu Hu [mhuam@connect.ust.hk].
