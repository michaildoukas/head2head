target_name=$1
dataset=head2headDataset # hardcoded
epoch=latest # hardcoded

python test.py --target_name $target_name --name head2head_$target_name --dataroot datasets/$dataset/dataset --which_epoch $epoch

results_dir=results/head2head_${target_name}/${epoch}_epoch/${dataset}_test/${target_name}

python scripts/images_to_video.py --results_dir $results_dir --output_mode all_heatmaps
