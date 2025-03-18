SCENE=$1

python render_volume.py --mode train --conf ./configs/womask_iron.conf --case ${SCENE}

python render_surface.py --data_dir /data2/jingzhi/IRON/data_flashlight/${SCENE}/train \
                        --out_dir ./PIR_stage2/${SCENE} \
                        --neus_ckpt_fpath ./PIR_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                        --inv_gamma_gt \
                        --patch_size 64 \
                        --num_iters 10001 \
                        --roughness_warmup_step 2001 \
                        --indirect_warmup_step 10001 \
                        --render_visibility \
                        --render_visibility_step 4001


python render_surface.py --data_dir /data2/jingzhi/IRON/data_flashlight/${SCENE}/train \
                        --out_dir ./PIR_stage2/${SCENE} \
                        --neus_ckpt_fpath ./PIR_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                        --inv_gamma_gt \
                        --patch_size 128 \
                        --num_iters 50002 \
                        --roughness_warmup_step 2001 \
                        --indirect_warmup_step 10001 \
                        --plot_image_name "6.png" \
                        --render_brdf