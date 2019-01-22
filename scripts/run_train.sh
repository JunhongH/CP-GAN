cartoon="/home/huangjunhong/datasets/emoji_data_align_white"
real_face="/home/huangjunhong/datasets/celeba_frontal_align"
test_dataset="./cartoon_test_data"
num_experiment=1
gan_weight=1
gan_local_weight=0.1
content=0.01
identity=0.01
lr=0.0001
content_network_name=light_CNN9
image_size=128
batch_size=1
epoch=10
log_interval=500
norm="instance"
n_layers_D=3
CUDA_VISIBLE_DEVICES=2 python src/neural_style.py train  \ 
 --n_layers_D ${n_layers_D} \ 
 --norm ${norm} \ 
 --dataset ${cartoon} \ 
 --test_dataset ${test_dataset} \ 
 --style_dataset  ${real_face} \ 
 --epochs ${epoch} \ 
 --cuda 1 \ 
 --log-interval ${log_interval} \ 
 --num_experiment ${num_experiment} \ 
 --gpu_ids 0 \ 
 --gan_weight ${gan_weight} \ 
 --gan_local_weight ${gan_local_weight} \ 
 --batch-size ${batch_size} \ 
 --content-weight ${content} \ 
 --lr ${lr} \ 
 --checkpoint \ 
 --identity_weight ${identity} \ 
 --image-size ${image_size} \ 
 # --resume_netG checkpoint/experiment_46/Epoch_62_iters_23820_G.model --resume_netD checkpoint/experiment_46/Epoch_62_iters_23820_D.model --start_epoch 63


