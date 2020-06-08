target_name=$1
dataset_name=$2
numseq=$3
epoch=latest # hardcoded

python test.py --checkpoints_dir checkpoints/$dataset_name \
               --results_dir results/$dataset_name \
               --target_name $target_name \
               --name head2head_${target_name}_${numseq} \
               --dataroot datasets/$dataset_name/dataset \
               --which_epoch $epoch

results_dir=results/$dataset_name/head2head_${target_name}_${numseq}/${epoch}_epoch/test/${target_name}

python scripts/images_to_video.py --results_dir $results_dir --output_mode source_target_separate
