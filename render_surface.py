import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import imageio
import cv2
imageio.plugins.freeimage.download()
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
import glob

from src.network import SDFNetwork, SingleVarianceNetwork, DINONetwork, BlendNetwork
from src.renderer.raytracer import RayTracer, Camera, render_camera, render_camera_visibility, dino_camera
from src.renderer.renderer_ggx import GGXColocatedRenderer
from src.models.extractor import VitExtractor
from src.models.image_losses import PyramidL2Loss, ssim_loss_fn
from src.blender.export_mesh import export_mesh
from src.blender.export_materials import export_materials
from src.evaluation import evaluation, eval_all

from src.utils import *
from src.parser import config_parser
from src.network import RenderingNetwork, PointLightNetwork
from src.loader import load_from_neus, load_dino_ckpt, load_pir_ckpt, load_dataset

dino_dimension=384

parser = config_parser()
args = parser.parse_args()
device = torch.device('cuda')

os.makedirs(args.out_dir, exist_ok=True)
parser.write_config_file(args, [os.path.join(args.out_dir, "args.txt"),])

###### rendering functions

def get_material_grads(specular, points):
    d_output = torch.ones_like(specular, requires_grad=False, device=specular.device)
    specular_grads = torch.autograd.grad(
        outputs=specular,
        inputs=points,
        grad_outputs=d_output,
        create_graph=True,
    )[0]
    return specular_grads


def get_materials(
    color_network_dict,
    points,
    normals,
    geo_f,
    dino_f,
    roughness_warmup=False,
    is_training=True,
    is_metal=args.is_metal,
):
    r"""get BRDF parameters:
        diffuse(3) + specular(3) + roughness(1) = 7
    """
    diffuse_albedo = color_network_dict["diffuse_albedo_network"](points, normals, -normals, geo_f, None).abs()[
        ..., [2, 1, 0]
    ]
    specular_albedo = color_network_dict["specular_albedo_network"](points, normals, None, geo_f, dino_f).abs()
    if not is_metal:
        specular_albedo = torch.mean(specular_albedo, dim=-1, keepdim=True).expand_as(specular_albedo)
    if not roughness_warmup:
        specular_roughness = color_network_dict["specular_roughness_network"](points, normals, None, geo_f, dino_f).abs() + 0.01
    else:
        specular_roughness = torch.ones((specular_albedo.shape[0], 1), device=points.device) * 0.10 # fixed roughness for warmup
    
    # specular_grads, roughness_grads for smoothness loss
    if args.smoothness_weight > 0 and is_training and not roughness_warmup:
        specular_grads = get_material_grads(specular_albedo, points)
        roughness_grads = get_material_grads(specular_roughness, points)
    else:
        specular_grads = torch.zeros_like(specular_albedo, device=specular_albedo.device)
        roughness_grads = torch.zeros_like(specular_roughness[..., 0:1], device=specular_roughness.device) # [N, 1]
    ###### test #####
    # specular_albedo = torch.tensor([0.990, 0.855, 0.270], device=points.device).expand(points.shape)
    # specular_roughness = torch.tensor([0.055], device=points.device).expand((points.shape[0], 1))
    # specular_grads = torch.zeros_like(specular_albedo, device=specular_albedo.device)
    return diffuse_albedo, specular_albedo, specular_roughness, specular_grads, roughness_grads


def render_fn(
    interior_mask,
    color_network_dict,
    ray_o,
    ray_d,
    points,
    normals,
    geo_f,
    dino_f,
    is_training=True,
    light=None,
    lightdir=None,
    distance=None,
    roughness_warmup=False,
    EDirectRadiance=True,
):
    r"""
        render function, works only when interior_mask is provided.
        return (diffuse_color, specular_color, diffuse_albedo, specular_albedo, specular_roughness, normal)
    """
    dots_sh = list(interior_mask.shape)
    rgb = torch.zeros(dots_sh + [3,], dtype=torch.float32, device=interior_mask.device)
    
    diffuse_rgb = rgb.clone()
    specular_rgb = rgb.clone()
    diffuse_albedo = rgb.clone()
    specular_albedo = rgb.clone()
    specular_grads = rgb.clone()
    roughness_grads = rgb.clone()
    specular_roughness = rgb[..., 0].clone()
    normals_pad = rgb.clone()
    if interior_mask.any():
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        interior_diffuse_albedo, interior_specular_albedo, interior_specular_roughness, \
        interior_specular_grads, interior_roughness_grads = get_materials(
            color_network_dict, points, normals, geo_f, dino_f,
            is_training=is_training, roughness_warmup=roughness_warmup,
        )
        
        if light is None:
            light = color_network_dict["point_light_network"]()
        
        if lightdir is None:
            lightdir = ray_d # co-located setting
            
        if distance is None:
            distance = (points - ray_o).norm(dim=-1, keepdim=True)
        
        results = ggx_renderer_modified(
            light,
            distance,
            # (points - ray_o).norm(dim=-1, keepdim=True),
            normals,
            -lightdir,
            -ray_d,
            interior_diffuse_albedo,
            interior_specular_albedo,
            interior_specular_roughness,
            EDirectRadiance=EDirectRadiance,
        )
        
        rgb[interior_mask] = results["rgb"]
        diffuse_rgb[interior_mask] = results["diffuse_rgb"]
        specular_rgb[interior_mask] = results["specular_rgb"]
        diffuse_albedo[interior_mask] = interior_diffuse_albedo
        specular_albedo[interior_mask] = interior_specular_albedo
        specular_grads[interior_mask] = interior_specular_grads
        roughness_grads[interior_mask] = interior_roughness_grads
        specular_roughness[interior_mask] = interior_specular_roughness.squeeze(-1)
        normals_pad[interior_mask] = normals

    render_res = {
        "color": rgb,
        "diffuse_color": diffuse_rgb,
        "specular_color": specular_rgb,
        "diffuse_albedo": diffuse_albedo,
        "specular_albedo": specular_albedo,
        "specular_grads": specular_grads,
        "specular_roughness": specular_roughness,
        "roughness_grads": roughness_grads,
        "normal": normals_pad,
    }
    if dino_f is not None:
        dino = torch.zeros(dots_sh + [dino_dimension,], dtype=torch.float32, device=interior_mask.device)
        if interior_mask.any():
            dino[interior_mask] = dino_f
        render_res.update({"dino_feature": dino})
    return render_res




image_fpaths, gt_images, Ks, W2Cs, grad_images = load_dataset(args.data_dir, train=(not args.render_all)) # load dataset
cameras = [
    Camera(W=gt_images[i].shape[1], H=gt_images[i].shape[0], K=Ks[i].cuda(), W2C=W2Cs[i].cuda())
    for i in range(gt_images.shape[0])
]
H, W = gt_images.shape[1:3]
ic(len(image_fpaths), gt_images.shape, Ks.shape, W2Cs.shape, len(cameras))


###### network specifications
with open(args.config_path, 'r') as f:
    model_config = json.load(f)

enable_offset = True

raytracer = RayTracer()
dino = VitExtractor(model_name='dino_vits8', device=device, usev2=False)

sdf_network = SDFNetwork(**model_config["sdf_network"]).cuda()
deviation_network = SingleVarianceNetwork(init_val=0.3).cuda()
inv_s = None
color_network_dict = {
    "diffuse_albedo_network": RenderingNetwork(**model_config["diffuse_albedo_network"]),
    "specular_albedo_network": RenderingNetwork(**model_config["specular_albedo_network"]),
    "specular_roughness_network": RenderingNetwork(**model_config["specular_roughness_network"]),
    "point_light_network": PointLightNetwork(enable_offset=enable_offset),
    "blend_network": BlendNetwork(**model_config["blend_network"]),
}
for key, network in color_network_dict.items():
    color_network_dict[key] = network.cuda()

dino_network = DINONetwork(**model_config["dino_network"]).cuda()

###### optimizer specifications
optimizer_config = model_config.get("optimizer")

sdf_lr = optimizer_config["sdf_network"]["lr"]
sdf_optimizer = torch.optim.Adam(sdf_network.parameters(), lr=sdf_lr)

point_light_paras = [{"params": [color_network_dict["point_light_network"].light], "lr": 2e-5}]
if args.render_visibility:
    point_light_paras.append({"params": list(color_network_dict["point_light_network"].light_offset.parameters()), "lr": 1e-4})

color_optimizer_dict = {}
for network_name, network in color_network_dict.items():
    if network_name == "point_light_network":
        color_optimizer_dict[network_name] = torch.optim.Adam(point_light_paras)
    else:
        lr = optimizer_config[network_name]["lr"]
        color_optimizer_dict[network_name] = torch.optim.Adam(network.parameters(), lr=lr)

dino_lr = optimizer_config["dino_network"]["lr"]
dino_optimizer = torch.optim.Adam(dino_network.parameters(), lr=dino_lr)

###### loss specifications
ggx_renderer_modified = GGXColocatedRenderer(use_cuda=True)
pyramidl2_loss_fn = PyramidL2Loss(use_cuda=True)

###### initialization using neus
load_from_neus(args.neus_ckpt_fpath, sdf_network, color_network_dict, deviation_network)
inv_s = deviation_network(torch.zeros([1, 3]).cuda())[:, :1].clip(1e-6, 1e6).detach()

dist = np.median([torch.norm(cameras[i].get_camera_origin()).item() for i in range(len(cameras))])
init_light = args.init_light_scale * dist * dist
color_network_dict["point_light_network"].set_light(init_light)

#### load pretrained checkpoints (stage2)
start_step = -1
dino_start_step = -1
out_dir = args.out_dir

ckpt_dino_fpaths = glob.glob(os.path.join(out_dir, "dino_*.pth"))
dino_start_step = load_dino_ckpt(ckpt_dino_fpaths, dino_network)

ckpt_fpaths = glob.glob(os.path.join(out_dir, "ckpt_*.pth"))
start_step = load_pir_ckpt(ckpt_fpaths, sdf_network, color_network_dict)

ic(dist, color_network_dict["point_light_network"].light.data)
if enable_offset:
    ic(color_network_dict["point_light_network"].light_offset.weight.data)
    # color_network_dict["point_light_network"].light_offset.weight.data = nn.Parameter(torch.tensor([[0.40, 0.0, 0.0]])).cuda()
ic(start_step)


###### export mesh and materials (export only)
blender_fpath = "/home/jingzhi/blender-4.1.1-linux-x64/blender"
if not os.path.isfile(blender_fpath):
    assert False, "blender is not installed!" # comment this line to install blender
    os.system(
        "wget https://mirror.clarkson.edu/blender/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz && \
             tar -xvf blender-4.1.1-linux-x64.tar.xz"
    )

def export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict):
    print(f"Exporting mesh and materials to: {export_out_dir}")
    sdf_fn = lambda x: sdf_network(x)[..., 0]
    print("Exporting mesh and uv...")
    with torch.no_grad():
        export_mesh(sdf_fn, os.path.join(export_out_dir, "mesh.obj"))
        os.system(
            f"{blender_fpath} --background --python src/blender/export_uv.py {os.path.join(export_out_dir, 'mesh.obj')} {os.path.join(export_out_dir, 'mesh.obj')}"
        )
    
    class MaterialPredictor(nn.Module):
        def __init__(self, sdf_network, color_network_dict):
            super().__init__()
            self.sdf_network = sdf_network
            self.color_network_dict = color_network_dict
            self.material_scale = color_network_dict["point_light_network"].get_light().item() / 10.0

        def forward(self, points):
            _, geo_f, normals = self.sdf_network.get_all(points, is_training=False)
            dino_f = dino_network(points, is_training=False)
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            diffuse_albedo, specular_albedo, specular_roughness, _, _ = get_materials(
                self.color_network_dict, points, normals, geo_f, dino_f, is_training=False
            )
            
            diffuse_albedo = diffuse_albedo * self.material_scale
            specular_albedo = specular_albedo * self.material_scale
            specular_roughness = specular_roughness
            return diffuse_albedo, specular_albedo, specular_roughness

    print("Exporting materials...")
    material_predictor = MaterialPredictor(sdf_network, color_network_dict)
    with torch.no_grad():
        export_materials(os.path.join(export_out_dir, "mesh.obj"), material_predictor, export_out_dir)

    print(f"Exported mesh and materials to: {export_out_dir}")


if args.export_all:
    export_out_dir = os.path.join(out_dir, f"mesh_and_materials_{start_step}")
    os.makedirs(export_out_dir, exist_ok=True)
    export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict)
    exit(0)


###### render all images (render only)
if args.render_all:
    render_out_dir = os.path.join(out_dir, f"render_{os.path.basename(args.data_dir)}_{start_step}")
    ic(f"Rendering images to: {render_out_dir}")
    
    os.makedirs(render_out_dir, exist_ok=True)
    n_cams = len(cameras)
    ic(n_cams)
    for i in tqdm.tqdm(range(n_cams)):
        cam, impath = cameras[i], image_fpaths[i]
        use_VQ = 0
        
        render_shadow = enable_offset
        if render_shadow:
            light_pos = cam.get_light_pos(color_network_dict["point_light_network"].get_offset())
        else:
            light_pos = None
        results = render_camera(
            cam,
            sdf_network,
            raytracer,
            color_network_dict,
            dino_network,
            render_fn,
            fill_holes=True,
            handle_edges=True,
            is_training=False,
            inv_s=inv_s,
            render_visibility=render_shadow,
            light_pos=light_pos,
        )
        color_im = results["color"]
        indirect_im = results["indirect_color"] * results["indirect_blend_coef"]
        # * render shadow
        if render_shadow:
            color_im = color_im * results["vis"][:, :, None]
        # * render indirect
        color_im = color_im + indirect_im
        
        if args.gamma_pred:
            color_im = torch.pow(color_im + 1e-6, 1.0 / 2.2)
        
        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()
        color_im = color_im.detach().cpu().numpy()
        
        
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0

        mask_im = results["convergent_mask"] | results["edge_mask"]
        maskpath = impath[:-4] + "_mask.png"
        os.makedirs(os.path.join(render_out_dir, "mask"), exist_ok=True)
        imageio.imwrite(os.path.join(render_out_dir, "mask", os.path.basename(maskpath)), to8b(mask_im)) # 保存 mask
        
        if args.render_normal:
            normalpath = impath[:-4] + "_normal.png"
            os.makedirs(os.path.join(render_out_dir, "normal"), exist_ok=True)
            np.save(os.path.join(render_out_dir, "normal", os.path.basename(normalpath)[:-4] + ".npy"), normal)
        
        if args.render_material:
            materials = ["diffuse_albedo", "specular_albedo", "specular_roughness"]
            for x in materials:
                # FIXME: export to video
                if 'albedo' in x:
                    light_intensity = color_network_dict["point_light_network"].get_light().item()
                    scale = light_intensity / 20.0
                else:
                    scale = 1.0
                x_im = results[x] * scale
                colorpath = impath[:-4] + ".exr"
                # colorpath = impath[:-4] + ".png"
                # x_im = to8b(np.power(x_im, 1.0 / 2.2))
                os.makedirs(os.path.join(render_out_dir, x), exist_ok=True)
                imageio.imwrite(os.path.join(render_out_dir, x, os.path.basename(colorpath)), x_im)
            
        colorpath = impath[:-4] + ".png"
        os.makedirs(os.path.join(render_out_dir, "rgb"), exist_ok=True)
        imageio.imwrite(os.path.join(render_out_dir, "rgb", os.path.basename(colorpath)), to8b(color_im))
        
        cleanup()
    
    # write light intensity
    light_intensity = color_network_dict["point_light_network"].get_light().item()
    with open(os.path.join(render_out_dir, "light.txt"), "w") as f:
        f.write(str(light_intensity))
    
    print("Evaluation start....")
    # evaluation(render_out_dir, args.eval_dir, use_mask=False)
    eval_all(render_out_dir, args.data_dir)
    exit(0)

##### training dino_network
fill_holes = False
dino_step_num = args.dino_step_num
dino_patch_size = 8
ratio = min(int(1024 / H), int(1024 / W))
dino_size_h, dino_size_w = H * ratio, W * ratio



def dino_feature(img):
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    img = F.interpolate(img, size=(dino_size_h, dino_size_w))
    _, _, h, w = img.shape
    with torch.no_grad():
        # if useDINOv2:
        #     dino_ret = dino.model.forward_features(img)['x_norm_patchtokens']
        # else:
        dino_ret = dino.get_vit_feature(img)[:, 1:, :]
        dino_ret = dino_ret.reshape([1, h//dino_patch_size, w//dino_patch_size, dino_dimension])
        # dino_ret = F.interpolate(dino_ret.permute(0, 3, 1, 2), size=(h, w)).permute(0, 2, 3, 1)
    return dino_ret.cpu().numpy()


def train_dino():
    dino_npy_path = glob.glob(os.path.join(out_dir, "dino_images.npy"))
    dino_images = []
    # load local ckpt
    if len(dino_npy_path) > 0:
        print("Load from dino_images.npy: ", dino_npy_path[0])
        dino_images = np.load(dino_npy_path[0], allow_pickle=True)
        dino_images = torch.from_numpy(dino_images)
    else:
        for i in tqdm.tqdm(range(len(gt_images)), desc='[Extracting DINO]', dynamic_ncols=True):
            dino_images.append(dino_feature(gt_images[i].cuda()))
            torch.cuda.empty_cache()
        dino_images = np.stack(dino_images)
        np.save(os.path.join(out_dir, "dino_images.npy"), dino_images)
        dino_images = torch.from_numpy(dino_images)

    loss_history = []

    pbar = tqdm.tqdm(range(dino_start_step+1, dino_step_num), desc="[Feature Init.]", dynamic_ncols=True)
    for dino_step in pbar:
        dino_optimizer.zero_grad()
        idx = np.random.randint(0, gt_images.shape[0])
        
        image_width = 200
        dino_image = F.interpolate(dino_images[idx].permute(0, 3, 1, 2), size=(H, W)).permute(0, 2, 3, 1).squeeze(0)
        camera_crop, crop_img_dict = cameras[idx].crop_region(
            trgt_W=image_width, trgt_H=image_width, images={"dino": dino_image}
        )
        gt_dino = crop_img_dict["dino"]
        results = dino_camera(camera_crop, sdf_network, raytracer, dino_network, is_training=True)
        surface_dino = results["dino"]
        mask = results["convergent_mask"]
        dino_loss = torch.Tensor([0.0]).cuda()
        
        if mask.any():
            surface_dino = surface_dino[mask]
            gt_dino = gt_dino.to(device)[mask]
            
            unreduced_dino = F.mse_loss(surface_dino, gt_dino, reduction="none")
            dino_loss = unreduced_dino.sum(dim=-1).nanmean()
        
            dino_loss.backward()
            dino_optimizer.step()
        
        if dino_step % 5000 == 0:
            torch.save(
                dict(
                    [("dino_network", dino_network.state_dict())]
                ),
                os.path.join(out_dir, f"dino_{dino_step}.pth")
            )
        
        if dino_loss > 0.:
            loss_history.append(dino_loss.item())
            if dino_step % 10 == 0:
                if len(loss_history) > 0:
                    pbar.set_postfix({"loss": sum(loss_history)/len(loss_history)})
                loss_history = []


def train_visibility(fill_holes=False, handle_edges=True):
    # print(f"light position: {color_network_dict['point_light_network'].get_offset}")
    vis_images = []
    fpath = os.path.join(out_dir, "vis_images.npy")
    if os.path.isfile(fpath):
        print(f"Load vis_images from neus checkpoint: {fpath}")
        vis_images = torch.from_numpy(np.load(os.path.join(out_dir, "vis_images.npy"), allow_pickle=True)).cuda()
    else:
        print("Rendering visibility...")
        for idx in tqdm.tqdm(range(len(cameras))):
            camera_vis = cameras[idx]
            results_vis = render_camera_visibility(
                camera_vis, sdf_network, raytracer,
                fill_holes=fill_holes,
                handle_edges=handle_edges,
                is_training=False,
                inv_s=inv_s,
                render_visibility=True,
                max_num_pts=2500,
                light_pos=camera_vis.get_light_pos(color_network_dict["point_light_network"].offset),
            )
            vis_images.append(results_vis["vis"])
            cleanup()
        vis_images = torch.stack(vis_images, dim=0).cuda()
        np.save(os.path.join(out_dir, f"vis_images.npy"), vis_images.cpu().numpy())
    return vis_images


if args.render_brdf:
    train_dino()


###### training
fill_holes = False
handle_edges = True
is_training = True
ssim_weight = args.ssim_weight
smoothness_weight = args.smoothness_weight
roughrange_weight = args.roughrange_weight
eik_weight = args.eik_weight
writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))
if args.inv_gamma_gt:
    ic("linearizing ground-truth images using inverse gamma correction")
    gt_images = torch.pow(gt_images, 2.2)


if args.render_brdf:
    vis_images = train_visibility()

global_step = None
for global_step in tqdm.tqdm(range(start_step + 1, args.num_iters), desc="[Train]", dynamic_ncols=True):
    sdf_optimizer.zero_grad()
    for x in color_optimizer_dict.keys():
        color_optimizer_dict[x].zero_grad()
    dino_optimizer.zero_grad()
    
    roughness_warmup = global_step < args.roughness_warmup_step
    indirect_warmup = global_step < args.indirect_warmup_step
    sdf_optim = not roughness_warmup
    
    render_shadow = False
    idx = np.random.randint(0, gt_images.shape[0])
    image_width = args.patch_size
    if args.render_visibility and roughness_warmup:
        image_width = 256 # for fast convergence
    
    if args.render_visibility: # ! invoked when render_visibility
        render_shadow = global_step >= args.render_visibility_step
    
    images = {
        "rgb": gt_images[idx],
        "grad": grad_images[idx],
    }
    if args.render_brdf:
        images["vis"] = vis_images[idx]
    camera_crop, crop_img_dict = cameras[idx].crop_region(
        trgt_W=image_width, trgt_H=image_width, images=images
    )
    gt_color_crop, gt_grad_crop, gt_vis_crop = crop_img_dict["rgb"], crop_img_dict["grad"], crop_img_dict.get("vis", None)
    
    # light position optimization
    if args.render_visibility:
        if render_shadow:
            light_pos = camera_crop.get_light_pos(color_network_dict["point_light_network"].offset)
        else:
            # fixed light position for brdf initialization
            light_pos = camera_crop.get_light_pos(torch.zeros((1, 3)).cuda())
    elif args.render_brdf:
        light_pos = camera_crop.get_light_pos(color_network_dict["point_light_network"].get_offset())
    else:
        light_pos = None

    results = render_camera(
        camera_crop,
        sdf_network,
        raytracer,
        color_network_dict,
        dino_network,
        render_fn,
        fill_holes=fill_holes,
        handle_edges=handle_edges,
        is_training=is_training,
        inv_s=inv_s,
        render_visibility=render_shadow,
        light_pos=light_pos,
        roughness_warmup=roughness_warmup,
        indirect_warmup=indirect_warmup,
    )
    # results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
    # results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)
    
    mask = results["convergent_mask"] # (128, 128)
    convergent_mask = mask.clone()
    if handle_edges:
        mask = mask | results["edge_mask"]

    img_loss = torch.Tensor([0.0]).cuda()
    img_l1_loss = torch.Tensor([0.0]).cuda()
    img_l2_loss = torch.Tensor([0.0]).cuda()
    img_ssim_loss = torch.Tensor([0.0]).cuda()
    roughrange_loss = torch.Tensor([0.0]).cuda()
    smoothness_loss = torch.Tensor([0.0]).cuda()
    
    # eikonal loss
    eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
    eik_grad = sdf_network.gradient(eik_points).view(-1, 3)
    eik_cnt = eik_grad.shape[0]
    eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).sum()
    # optimize covergent points
    if mask.any():
        pred_img = results["color"].permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        # FIXME: for stability
        # * gamma correction after synthesis
        if args.gamma_pred:
            pred_img = torch.pow(pred_img + 1e-6, 1.0 / 2.2)
        # * calculate shadow loss
        if render_shadow:
            vis_img = results["vis"]
            pred_img = pred_img * vis_img.expand(pred_img.shape)
        elif args.render_brdf:
            pred_img = pred_img * gt_vis_crop.expand(pred_img.shape)
        # * calculate inter-reflection after shadow
        if not indirect_warmup:
            indirect_img = results["indirect_color"] * results["indirect_blend_coef"]
            indirect_img = indirect_img.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
            pred_img = pred_img + indirect_img
        
        overexposure_mask = pred_img > 1.
        # FIXME: handle overexposure
        pred_img[overexposure_mask] = pred_img[overexposure_mask] ** 0.05
        
        gt_img = gt_color_crop.permute(2, 0, 1).unsqueeze(0).to(pred_img.device)
        gt_grad_crop = gt_grad_crop.to(pred_img.device)
        gt_grad = gt_grad_crop[convergent_mask].reshape(-1, )

        img_l1_loss = torch.abs(pred_img - gt_img).mean()
        img_l2_loss = pyramidl2_loss_fn(pred_img, gt_img)
        img_ssim_loss = ssim_weight * ssim_loss_fn(pred_img, gt_img, mask.unsqueeze(0).unsqueeze(0))
        img_loss = img_l1_loss + img_l2_loss + img_ssim_loss


        eik_grad = results["normal"][mask]
        eik_cnt += eik_grad.shape[0]
        eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()
        if "edge_pos_neg_normal" in results:
            eik_grad = results["edge_pos_neg_normal"]
            eik_cnt += eik_grad.shape[0]
            eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

        roughness = results["specular_roughness"][mask]
        roughness = roughness[roughness > 0.5]
        if roughness.numel() > 0:
            roughrange_loss = (roughness - 0.5).mean() * roughrange_weight

        if smoothness_weight > 0.:
            specular_grads = results["specular_grads"][convergent_mask]
            roughness_grads = results["roughness_grads"][convergent_mask]
            smoothness_loss = ((specular_grads.norm(dim=-1) / 1.732 + roughness_grads.norm(dim=-1)) * (-gt_grad).exp()).mean() * smoothness_weight
    
    eik_loss = eik_loss / eik_cnt * eik_weight
    if args.render_visibility:
        offset_x, offset_y, offset_z = color_network_dict["point_light_network"].offset[0]
    elif args.render_brdf:
        offset_x, offset_y, offset_z = color_network_dict["point_light_network"].get_offset()[0]
    else:
        offset_x, offset_y, offset_z = 0.0, 0.0, 0.0
    loss = img_loss + eik_loss + roughrange_loss + smoothness_loss
    if args.render_visibility:
        loss = loss + torch.abs(offset_z) * 0.5
    loss.backward()
    
    ####### optimizer.step() ########
    if sdf_optim:
        sdf_optimizer.step()
    
    if roughness_warmup:
        # roughness warmup
        color_optimizer_dict["diffuse_albedo_network"].step()
        color_optimizer_dict["specular_albedo_network"].step()
    else:
        if render_shadow:
            color_optimizer_dict["point_light_network"].step()
        else:
            for x in color_optimizer_dict.keys():
                color_optimizer_dict[x].step()

    # --------------- logging part ---------------
    if global_step % 50 == 0: # tensorboard logging
        writer.add_scalar("loss/loss", loss, global_step)
        writer.add_scalar("loss/img_loss", img_loss, global_step)
        writer.add_scalar("loss/img_l2_loss", img_l2_loss, global_step)
        writer.add_scalar("loss/img_ssim_loss", img_ssim_loss, global_step)
        writer.add_scalar("loss/eik_loss", eik_loss, global_step)
        writer.add_scalar("loss/roughrange_loss", roughrange_loss, global_step)
        writer.add_scalar("loss/smoothness_loss", smoothness_loss, global_step)
        writer.add_scalar("light", color_network_dict["point_light_network"].get_light().item(), global_step)
        writer.add_scalar("offset_x", offset_x, global_step)
        writer.add_scalar("offset_y", offset_y, global_step)
        writer.add_scalar("offset_z", offset_z, global_step)

    if global_step % 5000 == 0:
        torch.save(
            dict(
                [("sdf_network", sdf_network.state_dict()),]
                + [(x, color_network_dict[x].state_dict()) for x in color_network_dict.keys()]
            ),
            os.path.join(out_dir, f"ckpt_{global_step}.pth"),
        )

    if global_step % 200 == 0: # logging
        light_intensity = color_network_dict["point_light_network"].get_light().item()
        ic(
            out_dir,
            global_step,
            image_width,
            sdf_optim,
            indirect_warmup,
            loss.item(),
            light_intensity,
        )
        if args.render_visibility or args.render_brdf:
            ic(offset_x.item(), offset_y.item(), offset_z.item())

    if global_step % 2000 == 0 or args.debug: # dump image
        for x in list(results.keys()):
            del results[x]

        idx = 0 # xmen: 90, maneki: 6 in paper
        if args.plot_image_name is not None:
            while idx < len(image_fpaths):
                if args.plot_image_name in image_fpaths[idx]:
                    break
                idx += 1
        
        if args.debug or global_step % 10000 == 0: # full size image
            camera_resize, gt_color_resize = cameras[idx], gt_images[idx]
        else:
            camera_resize, gt_color_resize = cameras[idx].resize(factor=0.25, image=gt_images[idx])
        
        if enable_offset:
            light_pos = camera_resize.get_light_pos(color_network_dict["point_light_network"].get_offset())
        else:
            light_pos = None
        
        # if args.debug:
        #     render_shadow = True
        results = render_camera(
            camera_resize,
            sdf_network,
            raytracer,
            color_network_dict,
            dino_network,
            render_fn,
            fill_holes=fill_holes,
            handle_edges=handle_edges,
            is_training=False,
            inv_s=inv_s,
            render_visibility=render_shadow,
            light_pos=light_pos,
            roughness_warmup=roughness_warmup,
            indirect_warmup=indirect_warmup,
        )
        if not indirect_warmup:
            indirect_color_im = results["indirect_color"] * results["indirect_blend_coef"]
            indirect_color_im_wocoef = results["indirect_color"].detach().cpu().numpy()
            results["color"] = results["color"] + indirect_color_im
            indirect_color_im = indirect_color_im.detach().cpu().numpy()
        else:
            indirect_color_im = np.zeros_like(gt_color_resize)
        
        if args.gamma_pred:
            results["color"] = torch.pow(results["color"] + 1e-6, 1.0 / 2.2)
            results["diffuse_color"] = torch.pow(results["diffuse_color"] + 1e-6, 1.0 / 2.2)
            results["specular_color"] = torch.clamp(results["color"] - results["diffuse_color"], min=0.0)
        
        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()
        
        gt_color_im = gt_color_resize.detach().cpu().numpy()
        color_im = results["color"]
        diffuse_color_im = results["diffuse_color"]
        specular_color_im = results["specular_color"]
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        mask_im = results["convergent_mask"]
        edge_mask_im = np.tile(results["edge_mask"][:, :, np.newaxis], (1, 1, 3))
        
        if args.debug:
            # light_intensity = color_network_dict["point_light_network"].get_light().item()
            # scale = light_intensity / 60.0
            scale = 1.0
            print(f"scale: {scale}")
            results["diffuse_albedo"] = results["diffuse_albedo"] * (scale * 1.0)
            results["specular_albedo"] = results["specular_albedo"] * (scale * 1.0)
        
        # gamma correction (material)
        diffuse_albedo_im = np.power(results["diffuse_albedo"], 1.0 / 2.2)
        specular_albedo_im = np.power(results["specular_albedo"], 1.0 / 2.2)
        specular_roughness_im = np.power(np.tile(results["specular_roughness"][:, :, np.newaxis], (1, 1, 3)), 1.0 / 2.2)
        if args.inv_gamma_gt:
            gt_color_im = np.power(gt_color_im + 1e-6, 1.0 / 2.2)
            color_im = np.power(color_im + 1e-6, 1.0 / 2.2)
            diffuse_color_im = np.power(diffuse_color_im + 1e-6, 1.0 / 2.2)
            specular_color_im = color_im - diffuse_color_im
        
        # render shadow
        if args.render_brdf and args.debug:
            # color_im = color_im * vis_images[idx].cpu().numpy()[:, :, np.newaxis]
            # color_im = color_im * results["vis"][:, :, np.newaxis]
            pass
                
        row1 = np.concatenate([      gt_color_im,          normal_im,     indirect_color_im], axis=1)
        row2 = np.concatenate([         color_im,   diffuse_color_im,     specular_color_im], axis=1)
        row3 = np.concatenate([diffuse_albedo_im, specular_albedo_im, specular_roughness_im], axis=1)
        im = np.concatenate((row1, row2, row3), axis=0)
        imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}.png"), to8b(im))
        
        if args.debug:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            diff_map = np.abs(color_im - gt_color_im).mean(2)
            diff_map = np.clip(diff_map, 0.0, 0.50) / 0.50
            diff_map_color = cm.jet(diff_map)  # Normalize to range [0, 1] for color mapping
            plt.imsave(os.path.join(out_dir, f'logim_{global_step}_diff.png'), diff_map_color)
        
        ### DEBUG
        if args.debug:
            if not indirect_warmup:
                imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_indirect.png"), to8b(indirect_color_im))
            imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_gt_color_im.png"), to8b(gt_color_im))
            imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_color_im.png"), to8b(color_im))
            imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_diffuse_im.png"), to8b(diffuse_albedo_im))
            imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_specular_im.png"), to8b(specular_albedo_im))
            imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}_roughness_im.png"), to8b(specular_roughness_im))
            # dino_im = results["dino_feature"]
            # dino_im = PCA_visual(dino_im, mask=mask_im, normalize=True)
        
        # render visibility picture
        # if render_shadow and (args.debug or global_step % 1000 == 0):
        #     ic(color_network_dict["point_light_network"].get_offset())
        #     camera_vis = cameras[idx]
        #     results_vis = render_camera(
        #         camera_vis, sdf_network, raytracer, color_network_dict, clip_network, rough_network, dino_network, render_fn,
        #         fill_holes=fill_holes,
        #         handle_edges=handle_edges,
        #         is_training=False,
        #         inv_s=inv_s,
        #         render_visibility=True,
        #         light_pos=camera_vis.get_light_pos(color_network_dict["point_light_network"].offset()),
        #     )

        #     for x in list(results_vis.keys()):
        #         results_vis[x] = results_vis[x].detach().cpu().numpy()
        #     visibility_im = np.tile(results_vis["vis"][:, :, np.newaxis], (1, 1, 3))
        #     imageio.imwrite(os.path.join(out_dir, f"logim_vis_{global_step}.png"), to8b(visibility_im))

        cleanup()


###### export mesh and materials
# if global_step is None:
#     global_step = start_step
# export_out_dir = os.path.join(out_dir, f"mesh_and_materials_{global_step}")
# os.makedirs(export_out_dir, exist_ok=True)
# export_mesh_and_materials(export_out_dir, sdf_network, color_network_dict, clip_network, use_VQ=global_step>args.vq_start_step)
# # write light to light.txt
# with open(os.path.join(export_out_dir, "light.txt"), "w") as f:
#     f.write(str(color_network_dict["point_light_network"].get_light().item()))


if __name__ == "__main__":
    pass