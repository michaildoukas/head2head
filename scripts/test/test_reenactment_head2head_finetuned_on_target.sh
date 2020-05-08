target_name=$1

python test.py --target_name $target_name --name head2head_finetuned_$target_name --do_reenactment

python scripts/images_to_video.py --model_name head2head_finetuned_$target_name --output_mode source_target
