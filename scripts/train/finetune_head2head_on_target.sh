target_name=$1

python train.py --target_name $target_name --name head2head_finetuned_$target_name \
--load_pretrain checkpoints/head2head_all
