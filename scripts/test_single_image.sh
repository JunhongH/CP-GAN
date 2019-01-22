CUDA_VISIBLE_DEVICES=3 python use_light_AE.py test \
--dataset './test/318.jpg' \
--load_model './test/use_l2.pth' \
--save_address './test/' \
--use_fc 1 \
--ndf 128 \
--image_size 128 \