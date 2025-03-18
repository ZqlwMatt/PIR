import configargparse

def config_parser():
    r"""python render_surface.py --data_dir ./data_flashlight/${SCENE}/train \
                                 --out_dir ./exp_iron_stage2/${SCENE} \
                                 --neus_ckpt_fpath ./exp_iron_stage1/${SCENE}/checkpoints/ckpt_100000.pth \
                                 --num_iters 50001 --gamma_pred"""
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/model_config.json", help="path to model configuration file")
    parser.add_argument("--data_dir", type=str, default=None, help="input data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--neus_ckpt_fpath", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--num_iters", type=int, default=50001, help="number of iterations")
    parser.add_argument("--dino_step_num", type=int, default=10001, help="number of iterations for dino training")
    parser.add_argument("--patch_size", type=int, default=128, help="width and height of the rendered patches")
    parser.add_argument("--eik_weight", type=float, default=0.1, help="weight for eikonal loss")
    parser.add_argument("--ssim_weight", type=float, default=1.0, help="weight for ssim loss")
    parser.add_argument("--roughrange_weight", type=float, default=0.1, help="weight for roughness range loss")
    parser.add_argument("--smoothness_weight", type=float, default=0.0001, help="weight for smoothness loss")
    parser.add_argument("--roughness_warmup_step", type=int, default=2000, help="start iterations for roughness warmup")
    parser.add_argument("--indirect_warmup_step", type=int, default=10000, help="start iterations for indirect warmup")
    parser.add_argument("--render_visibility_step", type=int, default=4001, help="start iterations for rendering visibility")
    parser.add_argument("--render_material", default=False, action="store_true", help="whether to render materials")

    parser.add_argument("--plot_image_name", type=str, default=None, help="image to plot during training")
    parser.add_argument(
        "--inv_gamma_gt", action="store_true", help="whether to inverse gamma correct the ground-truth photos"
    )
    parser.add_argument("--gamma_pred", action="store_true", help="whether to gamma correct the predictions")
    parser.add_argument(
        "--is_metal", default=False, action="store_true",
        help="whether the object of interest is made of metals or the scene contains metals",
    )
    parser.add_argument("--init_light_scale", type=float, default=15.0, help="scaling parameters for light")
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="whether to export meshes and uv textures",
    )
    parser.add_argument(
        "--render_all",
        action="store_true",
        help="whether to render the input image set",
    )
    parser.add_argument("--render_normal", action="store_true", help="render the normal picture")
    parser.add_argument("--render_visibility", action="store_true", help="render with visibility")
    parser.add_argument("--render_brdf", action="store_true", help="render with BRDF (after light optimization)")
    parser.add_argument("--debug", default=False, action="store_true", help="enable debug mode, visualize images for paper")
    return parser

