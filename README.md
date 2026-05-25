# Overview
This is the first version of World4Drive (End-to-End Autonomous Driving via Intention-aware Physical Latent World Model).


# Step-by-step conda environment installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n w4d python=3.8 -y
conda activate w4d
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install -r requirements.txt
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
# get error using the above command, and use the below one
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install timm.**
```shell
pip install timm
```

**f. Install mmdet3d.**
```shell
conda activate w4d
git clone https://github.com/open-mmlab/mmdetection3d.git
cd /path/to/mmdetection3d
git checkout -f v0.17.1
# install an older version cudatoolkit in the conda env for the compile
conda install -c conda-forge cudatoolkit-dev=11.1.1
# and I use the pip because of a lot of dependency errors
pip install -v -e .
# python setup.py develop
```

**g. Install nuscenes-devkit.**
```shell
pip install nuscenes-devkit==1.1.9
pip install yapf==0.40.1
```

**h. Install other dependencies.**
```bash
pip install -r requirements.txt
# missing einops, mmengine
pip install einops
pip install mmengine
```

# Data preparation instructions

**Follow VAD to organize nuscenes and download the pickle**
```shell
$WORK_DIR/data/nuscenes/
```
For details, please refer to docs/prepare_dataset.md

# Train & Test

# Training
```shell
  ./tools/nusc_my_train.sh w4d/default 8
  # it's strange that i need to manually kill the processes as they are occupying the GPU memory after the crtl+c
  # pkill -u zzhao43 -9 -f python
  # tensorboard --logdir=work_dirs/test/tf_logs/ --port=6012
```

# Testing
```shell
  # ./tools/dist_test.sh $CONFIG $CKPT $NUM_GPU
  ./tools/dist_test.sh w4d/default work_dirs/test/epoch_1.pth 4
```