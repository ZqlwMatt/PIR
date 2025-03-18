import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import traceback


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print("Load data: Begin")
        self.device = torch.device("cuda")
        self.conf = conf

        self.data_dir = conf.get_string("data_dir")
        self.render_cameras_name = conf.get_string("render_cameras_name")
        self.object_cameras_name = conf.get_string("object_cameras_name")

        self.camera_outside_sphere = conf.get_bool("camera_outside_sphere", default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)  # not used

        import json

        camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict_norm.json")))
        for x in list(camera_dict.keys()):
            x = x[:-4] + ".png"
            camera_dict[x]["K"] = np.array(camera_dict[x]["K"]).reshape((4, 4))
            camera_dict[x]["W2C"] = np.array(camera_dict[x]["W2C"]).reshape((4, 4))

        self.camera_dict = camera_dict
        try:
            base_dir_len = len(os.path.join(self.data_dir, "image/"))
            self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*.png")), key=lambda x: int(x[base_dir_len:-4]))
            self.n_images = len(self.images_lis)
            self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 255.0
        except:
            print("Loading png images failed; try loading exr images")
            import pyexr

            self.images_lis = sorted(glob(os.path.join(self.data_dir, "image/*.exr")))
            self.n_images = len(self.images_lis)
            self.images_np = np.clip(
                np.power(np.stack([pyexr.open(im_name).get()[:, :, ::-1] for im_name in self.images_lis]), 1.0 / 2.2),
                0.0,
                1.0,
            )

        no_mask = True
        if no_mask:
            print("Not using masks")
            self.masks_lis = None
            self.masks_np = np.ones_like(self.images_np)
        else:
            try:
                self.masks_lis = sorted(glob(os.path.join(self.data_dir, "mask/*.png")))
                self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0
            except:
                # traceback.print_exc()
                print("Loading mask images failed; try not using masks")
                self.masks_lis = None
                self.masks_np = np.ones_like(self.images_np)

        self.images_np = self.images_np[..., :3]
        self.masks_np = self.masks_np[..., :3]

        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []
        self.world_mats_np = []
        for x in self.images_lis:
            x = os.path.basename(x)[:-4] + ".png"
            K = self.camera_dict[x]["K"].astype(np.float32)
            W2C = self.camera_dict[x]["W2C"].astype(np.float32)
            C2W = np.linalg.inv(self.camera_dict[x]["W2C"]).astype(np.float32)
            
            self.intrinsics_all.append(torch.from_numpy(K))
            self.pose_all.append(torch.from_numpy(C2W))
            self.world_mats_np.append(W2C)

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).to(self.device)
        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device)
        print("image shape, mask shape: ", self.images.shape, self.masks.shape)
        print("image pixel range: ", self.images.min().item(), self.images.max().item())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        # Bounding Box limits
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        object_scale_mat = np.eye(4).astype(np.float32)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print("Load data: End")

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        # pixel coordinates -> camera coordinates -> world coordinates
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        # [1, 1, 3, 3] @ [W, H, 3, 1] -> [W, H, 3, 3] @ [W, H, 3, 1]
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape) # (rays_v, rays_o)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        return (rays_o, rays_v, color, mask), size = (batch_size, 10)
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between `two` cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()

        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)

        # trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)

        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()

        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()

        rays_o = trans[None, None, :3].expand(rays_v.shape)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        """
            return (near, far) of input rays
        """
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        if self.images_lis[idx].endswith(".exr"):
            import pyexr

            img = np.power(pyexr.open(self.images_lis[idx]).get()[:, :, ::-1], 1.0 / 2.2) * 255.0
        else:
            img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)

