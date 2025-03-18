import os
import traceback
import torch
import imageio.v2 as imageio
import cv2
import numpy as np
import json

path2step = lambda x: int(os.path.basename(x)[5 : -4]) # path -> step #

def load_from_neus(neus_ckpt_fpath, sdf_network, color_network_dict, deviation_network):
    if os.path.isfile(neus_ckpt_fpath):
        print(f"Loading from neus checkpoint: {neus_ckpt_fpath}")
    ckpt = torch.load(neus_ckpt_fpath, map_location=torch.device("cuda"))
    sdf_network.load_state_dict(ckpt["sdf_network_fine"])
    color_network_dict["diffuse_albedo_network"].load_state_dict(ckpt["color_network_fine"])
    deviation_network.load_state_dict(ckpt["variance_network_fine"])


def load_dino_ckpt(dino_ckpt_fpaths, dino_network):
    if len(dino_ckpt_fpaths) > 0:
        dino_ckpt_fpaths = sorted(dino_ckpt_fpaths, key=path2step)
        
        dino_ckpt_fpath = dino_ckpt_fpaths[-1]
        dino_step = path2step(dino_ckpt_fpath)
        print("Loading from checkpoint(dino): ", dino_ckpt_fpath)
        
        ckpt_dino = torch.load(dino_ckpt_fpath, map_location=torch.device("cuda"))
        dino_network.load_state_dict(ckpt_dino["dino_network"])
        return dino_step
    return -1


def load_pir_ckpt(ckpt_fpaths, sdf_network, color_network_dict):
    if len(ckpt_fpaths) > 0:
        ckpt_fpaths = sorted(ckpt_fpaths, key=path2step)
        ckpt_fpath = ckpt_fpaths[-1]
        start_step = path2step(ckpt_fpath)
        print("Loading from checkpoint: ", ckpt_fpath)
        ckpt = torch.load(ckpt_fpath, map_location=torch.device("cuda"))
        # load model parameters for sdf, color_networks [4]
        sdf_network.load_state_dict(ckpt["sdf_network"])
        for x in list(color_network_dict.keys()):
            print(f"Loading {x}...")
            color_network_dict[x].load_state_dict(ckpt[x])
        return start_step
    return -1


def load_dataset(datadir: str, train: bool = True):
    """
    Get `image_paths`, `gt images`, `Ks`, `W2Cs` from `cam_dict_norm.json`.
    """
    cam_dict = json.load(open(os.path.join(datadir, "cam_dict_norm.json")))
    imgnames = list(cam_dict.keys())
    try:
        imgnames = sorted(imgnames, key=lambda x: int(x[:-4]))
    except:
        imgnames = sorted(imgnames)

    image_fpaths = []
    gt_images = []
    Ks = []
    W2Cs = []
    grad_images = []
    
    for x in imgnames:
        fpath = os.path.join(datadir, "image", x)
        assert fpath[-4:] in [".jpg", ".png"], "must use ldr images as inputs"
        if train:
            im = imageio.imread(fpath).astype(np.float32) / 255.0
        else:
            im = np.ones([512, 512, 3], dtype=np.float32)
        K = np.array(cam_dict[x]["K"]).reshape((4, 4)).astype(np.float32)
        W2C = np.array(cam_dict[x]["W2C"]).reshape((4, 4)).astype(np.float32)

        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)                                    # [H, W]
        im_grad_x = cv2.Sobel(im_gray, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)         # [H, W]
        im_grad_y = cv2.Sobel(im_gray, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)         # [H, W]
        im_grad = cv2.magnitude(im_grad_x, im_grad_y)
        
        image_fpaths.append(fpath)
        gt_images.append(torch.from_numpy(im))
        Ks.append(torch.from_numpy(K))
        W2Cs.append(torch.from_numpy(W2C))
        grad_images.append(torch.from_numpy(im_grad))
        # vis_images.append(torch.from_numpy(im_vis))
    gt_images = torch.stack(gt_images, dim=0)
    Ks = torch.stack(Ks, dim=0)
    W2Cs = torch.stack(W2Cs, dim=0)
    return image_fpaths, gt_images, Ks, W2Cs, grad_images
