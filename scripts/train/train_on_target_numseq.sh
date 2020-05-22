target_name=$1
dataset_name=$2
numseq=$3

python train.py --checkpoints_dir checkpoints/$dataset_name \
                --target_name $target_name \
                --name head2head_${target_name}_${numseq} \
                --dataroot datasets/$dataset_name/dataset \
                --max_n_sequences $numseq
