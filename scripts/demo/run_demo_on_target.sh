target_name=$1
dataset_name=$2

# Using two GPUs (for ~18 fps at two Nvidia GeForce RTX 2080 Ti)

python demo.py --checkpoints_dir checkpoints/$dataset_name \
               --results_dir results/$dataset_name \
               --target_name $target_name \
               --name head2head_$target_name \
               --dataroot datasets/$dataset_name/dataset \
               --do_reenactment \
               --gpu_ids 0,1 \
               --demo_dir results_demo # Comment this line out for camera demo.
