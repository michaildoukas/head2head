source_name=$1
target_name=$2
dataset_name=$3
epoch=latest # hardcoded

python preprocessing/reenact.py --dataset_name $dataset_name \
                                --source_id $source_name \
                                --target_id $target_name \
                                --keep_target_pose

python test.py --checkpoints_dir checkpoints/$dataset_name \
               --results_dir results/$dataset_name \
               --target_name $target_name \
               --source_name $source_name \
               --name head2head_finetuned_$target_name \
               --dataroot datasets/$dataset_name/dataset \
               --which_epoch $epoch \
               --do_reenactment


results_dir=results/$dataset_name/head2head_finetuned_${target_name}/${epoch}_epoch/test/${target_name}_${source_name}

python scripts/images_to_video.py --results_dir $results_dir --output_mode source_target_separate
