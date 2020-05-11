target_name=$1
dataset=head2headDataset # hardcoded

python train.py --target_name $target_name --name head2head_finetuned_$target_name --dataroot datasets/$dataset/dataset \
--load_pretrain checkpoints/head2head_all
