target_name=$1

python test.py --target_name $target_name --name head2head_$target_name

python scripts/images_to_video.py --model_name head2head_$target_name --output_mode all_heatmaps
