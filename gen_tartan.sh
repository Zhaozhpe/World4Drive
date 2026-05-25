CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
TEST_DATA_PATH=${TEST_DATA_PATH:-tartan_drive.json}
DEPTH_OUTPUT_DIR=${DEPTH_OUTPUT_DIR:-data/tartan_drive_2.0/depth_gaussian}
BEGIN=${BEGIN:-0}
END=${END:-100000000}
BATCH_SIZE=${BATCH_SIZE:-1}

METRIC3D_DEPTH_OUTPUT_DIR="$DEPTH_OUTPUT_DIR" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \
    --load-from weight/metric_depth_vit_giant2_800k.pth \
    --test_data_path "$TEST_DATA_PATH" \
    --launcher None \
    --batch_size "$BATCH_SIZE" \
    --begin "$BEGIN" \
    --end "$END"
