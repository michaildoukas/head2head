target_name=$1

python test.py --target_name $target_name --name head2head_finetuned_$target_name

python scripts/images_to_video.py --name head2head_finetuned_$target_name
