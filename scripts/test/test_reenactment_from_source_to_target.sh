source_name=$1
target_name=$2
dataset=$3
epoch=latest # hardcoded

python preprocessing/reenact.py --dataset_name $dataset --source_id $source_name --target_id $target_name

python test.py --target_name $target_name --source_name $source_name --name head2head_$target_name --do_reenactment --dataroot datasets/$dataset/dataset --which_epoch $epoch

results_dir=results/head2head_${target_name}/${epoch}_epoch/${dataset}_test/${target_name}_${source_name}

python scripts/images_to_video.py --results_dir $results_dir --output_mode source_target
