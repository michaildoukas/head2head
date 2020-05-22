target_name=$1
dataset_name=$2
numseq=$3

python train.py --checkpoints_dir checkpoints/$dataset_name \
                --target_name $target_name \
                --name head2head_finetuned_${target_name}_${numseq} \
                --dataroot datasets/$dataset_name/dataset \
                --load_pretrain checkpoints/faceforensicspp/head2head_faceforensicspp \
                --max_n_sequences $numseq
