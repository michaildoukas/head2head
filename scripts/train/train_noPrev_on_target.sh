target_name=$1
dataset_name=$2
gpu=$3

nohup python train.py --checkpoints_dir checkpoints/$dataset_name \
                --target_name $target_name \
                --name head2head_noPrev_$target_name \
                --dataroot datasets/$dataset_name/dataset \
                --no_prev_output \
                --gpu_ids $gpu >/dev/null 2>&1 &
