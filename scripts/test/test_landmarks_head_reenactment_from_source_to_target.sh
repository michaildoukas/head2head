source_name=$1
target_name=$2
dataset_name=$3
epoch=latest # hardcoded

mkdir -p datasets/$dataset_name/dataset/test/source_landmarks70/${target_name}_${source_name}
mkdir -p datasets/$dataset_name/dataset/test/source_images/${target_name}_${source_name}
cp datasets/$dataset_name/dataset/test/landmarks70/${source_name}/* datasets/$dataset_name/dataset/test/source_landmarks70/${target_name}_${source_name}
cp datasets/$dataset_name/dataset/test/images/${source_name}/* datasets/$dataset_name/dataset/test/source_images/${target_name}_${source_name}

python test.py --checkpoints_dir checkpoints/$dataset_name \
               --results_dir results/$dataset_name \
               --target_name $target_name \
               --source_name $source_name \
               --name head2head_landmarks_$target_name \
               --dataroot datasets/$dataset_name/dataset \
               --which_epoch $epoch \
               --do_reenactment \
               --dataset_mode landmarks --no_eye_gaze --input_nc 3

results_dir=results/$dataset_name/head2head_landmarks_${target_name}/${epoch}_epoch/test/${target_name}_${source_name}

python scripts/images_to_video.py --results_dir $results_dir --output_mode source_nmfc_target
