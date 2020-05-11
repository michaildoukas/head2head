target_name=$1
dataset=$2

python train.py --target_name $target_name --name head2head_$target_name --dataroot datasets/$dataset/dataset
