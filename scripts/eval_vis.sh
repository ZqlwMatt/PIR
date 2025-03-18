SCENE=$1

python render_surface.py --data_dir /data2/jingzhi/IRON/data_flashlight/${SCENE}/test \
                        --out_dir ./exp_iron_stage2/${SCENE} \
                        --neus_ckpt_fpath ./exp_iron_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                        --gamma_pred \
                        --patch_size 128 \
                        --num_iters 60001 \
                        --roughness_warmup_step 2001 \
                        --indirect_warmup_step 12001 \
                        --smoothness_weight 0.0001 \
                        --init_light_scale 15.0 \
                        --render_brdf \
                        --render_all \
                        --render_material
