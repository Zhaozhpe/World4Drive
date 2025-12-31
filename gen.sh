CUDA_VISIBLE_DEVICES=7 python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \
    --load-from weight/metric_depth_vit_giant2_800k.pth \
    --test_data_path nuscenes_val.json \
    --launcher None \
    --batch_size 1 \
    --begin 0 \
    --end 100

# CUDA_VISIBLE_DEVICES=0 python mono/tools/test_scale_cano.py \
#     'mono/configs/HourglassDecoder/convtiny.0.3_150.py' \
#     --load-from weight/convtiny_hourglass_v1.pth \
#     --test_data_path nuscenes_train.json \
#     --launcher None \
#     --batch_size 1 \
#     --begin 0 \
#     --end 200000

# CUDA_VISIBLE_DEVICES=3 python mono/tools/test_scale_cano.py \
#     'mono/configs/HourglassDecoder/vit.raft5.small.py' \
#     --load-from weight/metric_depth_vit_small_800k.pth \
#     --test_data_path nuscenes_val.json \
#     --launcher None \
#     --batch_size 24 \
#     --begin 0 \
#     --end 200000