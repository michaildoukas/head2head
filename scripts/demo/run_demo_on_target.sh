target_name=$1
dataset_name=$2

python demo.py --checkpoints_dir checkpoints/$dataset_name \
               --results_dir results/$dataset_name \
               --target_name $target_name \
               --name head2head_$target_name \
               --dataroot datasets/$dataset_name/dataset \
               --do_reenactment \
               --demo_dir demo
