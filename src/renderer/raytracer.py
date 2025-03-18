from email import contentmanager
from operator import contains
import os
from sys import prefix
from turtle import update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia
import cv2

from icecream import ic

VERBOSE_MODE = False
dimension = 384

def reparam_points(nondiff_points, nondiff_grads, nondiff_trgt_dirs, diff_sdf_vals):
    # note that flipping the direction of nondiff_trgt_dirs would not change this equations at all
    # hence we require dot >= 0
    dot = (nondiff_grads * nondiff_trgt_dirs).sum(dim=-1, keepdim=True)
    # assert (dot >= 0.).all(), 'dot>=0 not satisfied in reparam_points: {},{}'.format(dot.min().item(), dot.max().item())
    dot = torch.clamp(dot, min=1e-4)
    diff_points = nondiff_points - nondiff_trgt_dirs / dot * (diff_sdf_vals - diff_sdf_vals.detach()) # detach to avoid double grad
    return diff_points


class RayTracer(nn.Module):
    r"""
        Usage: forward(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask)
        
        return dict: {
            'convergent_mask': convergent_mask,
            'points': curr_start_points,
            'sdf': curr_start_sdf,
            'distance': acc_start_dis
        }
    """
    def __init__(
        self,
        sdf_threshold=5.0e-5,
        sphere_tracing_iters=16,
        n_steps=128,
        max_num_pts=200000,
    ):
        super().__init__()
        """sdf values of convergent points must be inside [-sdf_threshold, sdf_threshold]"""
        self.sdf_threshold = sdf_threshold
        # sphere tracing hyper-params
        self.sphere_tracing_iters = sphere_tracing_iters
        # dense sampling hyper-params
        self.n_steps = n_steps

        self.max_num_pts = max_num_pts

    @torch.no_grad()
    def forward(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask):
        """
            Finds intersection points between rays and the SDF surface using sphere tracing and ray sampling.
            Returns:
                sdf: [..., 1]
                ray_o: [..., 3]
                ray_d: [..., 3]
                min_dis: [..., 1]
                max_dis: [..., 1]
                work_mask: [..., 1]
        """
        (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_start_sdf,
            acc_start_dis,
        ) = self.sphere_tracing(sdf, ray_o, ray_d, min_dis, max_dis, work_mask)
        sphere_tracing_cnt = convergent_mask.sum()

        sampler_work_mask = unfinished_mask_start
        sampler_cnt = 0
        if sampler_work_mask.sum() > 0:
            tmp_mask = (curr_start_sdf[sampler_work_mask] > 0.0).float()
            sampler_min_dis = (
                tmp_mask * acc_start_dis[sampler_work_mask] + (1.0 - tmp_mask) * min_dis[sampler_work_mask]
            )
            sampler_max_dis = (
                tmp_mask * max_dis[sampler_work_mask] + (1.0 - tmp_mask) * acc_start_dis[sampler_work_mask]
            )

            (sampler_convergent_mask, sampler_points, sampler_sdf, sampler_dis,) = self.ray_sampler(
                sdf,
                ray_o[sampler_work_mask],
                ray_d[sampler_work_mask],
                sampler_min_dis,
                sampler_max_dis,
            )

            convergent_mask[sampler_work_mask] = sampler_convergent_mask
            curr_start_points[sampler_work_mask] = sampler_points
            curr_start_sdf[sampler_work_mask] = sampler_sdf
            acc_start_dis[sampler_work_mask] = sampler_dis
            sampler_cnt = sampler_convergent_mask.sum()

        ret_dict = {
            "convergent_mask": convergent_mask,
            "points": curr_start_points,
            "sdf": curr_start_sdf,
            "distance": acc_start_dis,
        }

        if VERBOSE_MODE:  # debug
            sdf_check = sdf(curr_start_points)
            ic(
                convergent_mask.sum() / convergent_mask.numel(),
                sdf_check[convergent_mask].min().item(),
                sdf_check[convergent_mask].max().item(),
            )
            debug_info = "Total,raytraced,convergent(sphere tracing+dense sampling): {},{},{} ({}+{})".format(
                work_mask.numel(),
                work_mask.sum(),
                convergent_mask.sum(),
                sphere_tracing_cnt,
                sampler_cnt,
            )
            ic(debug_info)
        return ret_dict

    def sphere_tracing(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask):
        """
            Run sphere tracing algorithm for max iterations
            each iter, curr_start_poitns += ray_d * sdf(curr_start_points)
            return (
                convergent_mask,
                unfinished_mask_start,
                curr_start_points,
                curr_sdf_start,
                acc_start_dis
            )
        """
        iters = 0
        unfinished_mask_start = work_mask.clone()
        acc_start_dis = min_dis.clone()
        curr_start_points = ray_o + ray_d * acc_start_dis.unsqueeze(-1)
        curr_sdf_start = sdf(curr_start_points)
        while True:
            unfinished_mask_start = (
                unfinished_mask_start & (curr_sdf_start.abs() > self.sdf_threshold) & (acc_start_dis < max_dis)
            )

            if iters == self.sphere_tracing_iters or unfinished_mask_start.sum() == 0:
                break
            iters += 1

            # Make step
            sdfs = curr_sdf_start[unfinished_mask_start] # sdf of unfinished points
            acc_start_dis[unfinished_mask_start] += sdfs # \* ray norm
            curr_start_points[unfinished_mask_start] += ray_d[unfinished_mask_start] * sdfs.unsqueeze(-1)
            curr_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        convergent_mask = (
            work_mask
            & ~unfinished_mask_start
            & (curr_sdf_start.abs() <= self.sdf_threshold)
            & (acc_start_dis < max_dis)
        ) # convergent_mask ~= finished & (sdf < threshold) & (acc_dis < max_dis)
        return (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_sdf_start,
            acc_start_dis,
        )

    def ray_sampler(self, sdf, ray_o, ray_d, min_dis, max_dis):
        """Sample the ray in a given range and perform rootfinding on ray segments which have sign transition"""
        intervals_dis = (
            torch.linspace(0, 1, steps=self.n_steps).float().to(min_dis.device).view(1, self.n_steps)
        )  # [1, n_steps]
        intervals_dis = min_dis.unsqueeze(-1) + intervals_dis * (
            max_dis.unsqueeze(-1) - min_dis.unsqueeze(-1)
        )  # [n_valid, n_steps]
        points = ray_o.unsqueeze(-2) + ray_d.unsqueeze(-2) * intervals_dis.unsqueeze(-1)  # [n_valid, n_steps, 3]

        sdf_val = []
        for pnts in torch.split(points.reshape(-1, 3), self.max_num_pts, dim=0):
            sdf_val.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val, dim=0).reshape(-1, self.n_steps)

        # To be returned
        sampler_pts = torch.zeros_like(ray_d)
        sampler_sdf = torch.zeros_like(min_dis)
        sampler_dis = torch.zeros_like(min_dis)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).float().to(sdf_val.device).reshape(
            1, self.n_steps
        )
        # return first negative sdf point if exists
        min_val, min_idx = torch.min(tmp, dim=-1)
        rootfind_work_mask = (min_val < 0.0) & (min_idx >= 1)
        n_rootfind = rootfind_work_mask.sum()
        if n_rootfind > 0:
            # [n_rootfind, 1]
            min_idx = min_idx[rootfind_work_mask].unsqueeze(-1)
            z_low = torch.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(
                -1
            )  # [n_rootfind, ]
            # [n_rootfind, ]; > 0
            sdf_low = torch.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(-1)
            z_high = torch.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx).squeeze(
                -1
            )  # [n_rootfind, ]
            # [n_rootfind, ]; < 0
            sdf_high = torch.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx).squeeze(-1)

            p_pred, z_pred, sdf_pred = self.rootfind(
                sdf,
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                ray_o[rootfind_work_mask],
                ray_d[rootfind_work_mask],
            )

            sampler_pts[rootfind_work_mask] = p_pred
            sampler_sdf[rootfind_work_mask] = sdf_pred
            sampler_dis[rootfind_work_mask] = z_pred

        return rootfind_work_mask, sampler_pts, sampler_sdf, sampler_dis

    def rootfind(self, sdf, f_low, f_high, d_low, d_high, ray_o, ray_d):
        """binary search the root"""
        work_mask = (f_low > 0) & (f_high < 0)
        d_mid = (d_low + d_high) / 2.0
        i = 0
        while work_mask.any():
            p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
            f_mid = sdf(p_mid)
            ind_low = f_mid > 0
            ind_high = f_mid <= 0
            if ind_low.sum() > 0:
                d_low[ind_low] = d_mid[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if ind_high.sum() > 0:
                d_high[ind_high] = d_mid[ind_high]
                f_high[ind_high] = f_mid[ind_high]
            d_mid = (d_low + d_high) / 2.0
            work_mask &= (d_high - d_low) > 2 * self.sdf_threshold
            i += 1
        p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
        f_mid = sdf(p_mid)
        return p_mid, d_mid, f_mid


@torch.no_grad()
def intersect_sphere(ray_o, ray_d, r):
    """
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    return (mask, dist ||ray_d|| between two intersection points)
    """
    # note: d1 becomes negative if this mid point is behind camera
    # length of projection from ray origin to sphere center along ray direction
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    # intersection point of ray with perpendicular line from sphere center
    p = ray_o + d1.unsqueeze(-1) * ray_d
    
    # calculate squared difference between sphere radius and distance from p to sphere center
    # if tmp > 0, ray intersects sphere
    tmp = r * r - torch.sum(p * p, dim=-1)
    mask_intersect = tmp > 0.0  # rays that intersect the sphere
    # distance from p to intersection points
    d2 = torch.sqrt(torch.clamp(tmp, min=0.0)) / torch.norm(ray_d, dim=-1)
    
    return mask_intersect, torch.clamp(d1 - d2, min=0.0), d1 + d2


class Camera(object):
    def __init__(self, W, H, K, W2C):
        """
        W, H: int
        K, W2C: 4x4 tensor
        """
        self.W = W
        self.H = H
        self.K = K
        self.W2C = W2C
        self.K_inv = torch.inverse(K)
        self.C2W = torch.inverse(W2C)
        self.device = self.K.device
        # self.clip_scales = torch.ones((self.H, self.W)).cuda()
    
    def get_fx(self):
        return self.K[0, 0]
    
    def get_image_size(self):
        return self.H, self.W
    
    def get_light_pos(self, offset):
        """
        offset: [..., 3]
        return light position in world coordinate
        """
        return torch.matmul(offset, self.C2W[:3, :3].transpose(1, 0)) + self.C2W[:3, 3].unsqueeze(0)
        # return offset + self.C2W[:3,]

    def get_rays(self, uv):
        """
        Receives a set of coordinates in UV coordinate system (pixel coordinates) and calculates the corresponding rays.
        uv: [..., 2]
        """
        dots_sh = list(uv.shape[:-1])

        uv = uv.view(-1, 2)
        uv = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1)
        
        ray_d = torch.matmul(
            torch.matmul(uv, self.K_inv[:3, :3].transpose(1, 0)),
            self.C2W[:3, :3].transpose(1, 0)
        ).reshape(dots_sh + [3])

        ray_d_norm = ray_d.norm(dim=-1)
        ray_d = ray_d / ray_d_norm.unsqueeze(-1)

        # Set ray origins to camera position
        ray_o = self.C2W[:3, 3].unsqueeze(0).expand(uv.shape[0], -1).reshape(dots_sh + [3,])
        
        return ray_o, ray_d, ray_d_norm

    def get_camera_origin(self, prefix_shape=None):
        ray_o = self.C2W[:3, 3]
        if prefix_shape is not None:
            prefix_shape = list(prefix_shape)
            ray_o = ray_o.view([1,] * len(prefix_shape) + [3,]).expand(prefix_shape + [3,])
        return ray_o

    def get_uv(self):
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        uv = torch.from_numpy(np.stack((u, v), axis=-1).astype(np.float32)).to(self.device) + 0.5
        return uv

    def project(self, points):
        """
        Project points to camera space, uv: [..., 2]
        points: [..., 3]
        """
        dots_sh = list(points.shape[:-1])

        points = points.view(-1, 3)
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
        uv = torch.matmul(
            torch.matmul(points, self.W2C.transpose(1, 0)),
            self.K.transpose(1, 0),
        )
        uv = uv[:, :2] / uv[:, 2:3]

        uv = uv.view(dots_sh + [2,])
        return uv

    def crop_region(self, trgt_W, trgt_H, center_crop=False, ul_corner=None, images=None):
        r"""Crop image to size trgt_W, trgt_H
            images: dict of images to crop
            return (camera, cropped_images)
        """
        K = self.K.clone()
        if ul_corner is not None:
            ul_col, ul_row = ul_corner
        elif center_crop:
            ul_col = self.W // 2 - trgt_W // 2
            ul_row = self.H // 2 - trgt_H // 2
        else:
            ul_col = np.random.randint(0, self.W - trgt_W)
            ul_row = np.random.randint(0, self.H - trgt_H)
        K[0, 2] -= ul_col
        K[1, 2] -= ul_row

        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        cropped_images = {}
        if images is not None:
            for key, img in images.items():
                assert img.shape[0] == self.H and img.shape[1] == self.W, f"image size for {key} does not match specified size"
                cropped_images[key] = img[ul_row : ul_row + trgt_H, ul_col : ul_col + trgt_W]
        
        return camera, cropped_images

    def resize(self, factor, image=None):
        assert factor < 1.0
        trgt_H, trgt_W = int(self.H * factor), int(self.W * factor)
        K = self.K.clone()
        K[0, :3] *= trgt_W / self.W
        K[1, :3] *= trgt_H / self.H
        camera = Camera(trgt_W, trgt_H, K, self.W2C.clone())

        if image is not None:
            device = image.device
            resized_image = cv2.resize(
                image.detach().cpu().numpy(), 
                (trgt_W, trgt_H), 
                interpolation=cv2.INTER_AREA
            )
            image = torch.from_numpy(resized_image).to(device)
            
        return camera, image


@torch.no_grad()
def raytrace_pixels(sdf_network, raytracer, uv, camera, mask=None, max_num_rays=200000):
    """
        "convergent_mask": convergent_mask,
        "points": curr_start_points,
        "sdf": curr_start_sdf,
        "distance": acc_start_dis,
        "depth": acc_start_dis/ray_d_norm_split, # ?
        "uv": uv,
        "ray_o": ray_o,
        "ray_d": ray_d,
        "ray_d_norm": ray_d_norm,
        "clip_scales": clip_scales, or "cl"
        （(uv, rays, clip) 和 camera 对应）
    """
    if mask is None:
        mask = torch.ones_like(uv[..., 0]).bool()

    dots_sh = list(uv.shape[:-1])

    ray_o, ray_d, ray_d_norm = camera.get_rays(uv)
    sdf = lambda x: sdf_network(x)[..., 0]

    merge_results = None
    for ray_o_split, ray_d_split, ray_d_norm_split, mask_split in zip(
        torch.split(ray_o.view(-1, 3), max_num_rays, dim=0),
        torch.split(ray_d.view(-1, 3), max_num_rays, dim=0),
        torch.split(ray_d_norm.view(-1, ), max_num_rays, dim=0),
        torch.split(mask.view(-1, ), max_num_rays, dim=0),
    ):
        mask_intersect_split, min_dis_split, max_dis_split = intersect_sphere(ray_o_split, ray_d_split, r=1.0)
        results = raytracer(
            sdf,
            ray_o_split,
            ray_d_split,
            min_dis_split,
            max_dis_split,
            mask_intersect_split & mask_split,
        ) # 返回 convergent_mask, current points/sdf/acc_start_dis
        results["depth"] = results["distance"] / ray_d_norm_split # convert `distance` to `depth`

        # batch merge
        if merge_results is None:
            merge_results = dict(
                [
                    (
                        x,
                        [
                            results[x],
                        ],
                    )
                    for x in results.keys()
                    if isinstance(results[x], torch.Tensor)
                ]
            )
        else:
            for x in results.keys():
                merge_results[x].append(results[x])  # gpu
    # batch merge
    for x in list(merge_results.keys()):
        results = torch.cat(merge_results[x], dim=0).reshape(
            dots_sh
            + [
                -1,
            ]
        )
        if results.shape[-1] == 1:
            results = results[..., 0]
        merge_results[x] = results  # gpu

    merge_results.update(
        {
            "uv": uv,
            "ray_o": ray_o,
            "ray_d": ray_d,
            "ray_d_norm": ray_d_norm,
        }
    )
    
    return merge_results


def unique(x, dim=-1):
    """
    return: unique elements in x, and their original indices in x
    """
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


@torch.no_grad()
def locate_edge_points(
    camera, walk_start_points, sdf_network, max_step, step_size, dot_threshold, max_num_rays=200000, mask=None
):
    """walk on the surface to locate 3d edge points with high precision, run `alg 1.` in paper"""
    if mask is None:
        mask = torch.ones_like(walk_start_points[..., 0]).bool()

    walk_finish_points = walk_start_points.clone()
    walk_edge_found_mask = mask.clone()
    n_valid = mask.sum()
    if n_valid > 0:
        dots_sh = list(walk_start_points.shape[:-1])

        walk_finish_points_valid = []
        walk_edge_found_mask_valid = []
        # batch operation
        for cur_points_split in torch.split(walk_start_points[mask].clone().view(-1, 3).detach(), max_num_rays, dim=0):
            walk_edge_found_mask_split = torch.zeros_like(cur_points_split[..., 0]).bool()
            not_found_mask_split = ~walk_edge_found_mask_split # ones_like

            ray_o_split = camera.get_camera_origin(prefix_shape=cur_points_split.shape[:-1])

            i = 0
            while True: # walk on the surface until all batch points done or max_step reached
                cur_viewdir_split = ray_o_split[not_found_mask_split] - cur_points_split[not_found_mask_split]
                cur_viewdir_split = cur_viewdir_split / (cur_viewdir_split.norm(dim=-1, keepdim=True) + 1e-10) # o-x
                cur_sdf_split, _, cur_normal_split = sdf_network.get_all(
                    cur_points_split[not_found_mask_split].view(-1, 3),
                    is_training=False,
                )
                cur_normal_split = cur_normal_split / (cur_normal_split.norm(dim=-1, keepdim=True) + 1e-10)

                dot_split = (cur_normal_split * cur_viewdir_split).sum(dim=-1) # cos<normal, viewdir>
                tmp_not_found_mask = dot_split.abs() > dot_threshold
                walk_edge_found_mask_split[not_found_mask_split] = ~tmp_not_found_mask
                not_found_mask_split = ~walk_edge_found_mask_split

                if i >= max_step or not_found_mask_split.sum() == 0:
                    break
                # surface walk 的算法和论文的 alg 1. 不同
                cur_walkdir_split = cur_normal_split - cur_viewdir_split / dot_split.unsqueeze(-1) # n_t - (o - x_t)/((o - x_t) * n_t)
                cur_walkdir_split = cur_walkdir_split / (cur_walkdir_split.norm(dim=-1, keepdim=True) + 1e-10)
                # regularize walk direction such that we don't get far away from the zero iso-surface
                cur_walkdir_split = cur_walkdir_split - cur_sdf_split * cur_normal_split
                cur_points_split[not_found_mask_split] += (step_size * cur_walkdir_split)[tmp_not_found_mask] # step_size = 1e-3

                i += 1

            walk_finish_points_valid.append(cur_points_split)
            walk_edge_found_mask_valid.append(walk_edge_found_mask_split)
        # 保存终止点 和 found status
        walk_finish_points[mask] = torch.cat(walk_finish_points_valid, dim=0)
        walk_edge_found_mask[mask] = torch.cat(walk_edge_found_mask_valid, dim=0)
        walk_finish_points = walk_finish_points.reshape(
            dots_sh
            + [
                3,
            ]
        )
        walk_edge_found_mask = walk_edge_found_mask.reshape(dots_sh)
    # * ------------------------------ points batch vars
    edge_points = walk_finish_points[walk_edge_found_mask]
    edge_mask = torch.zeros(camera.H, camera.W).bool().to(walk_finish_points.device)
    edge_uv = torch.zeros_like(edge_points[..., :2])
    update_pixels = torch.Tensor([]).long().to(walk_finish_points.device)
    if walk_edge_found_mask.any():
        # * filter out edge points out of camera's fov;
        # * if there are multiple edge points mapping to the same pixel, only keep one
        edge_uv = camera.project(edge_points)
        update_pixels = torch.floor(edge_uv.detach()).long() # convert uv to pixel index
        update_pixels = update_pixels[:, 1] * camera.W + update_pixels[:, 0]
        mask = (update_pixels < camera.H * camera.W) & (update_pixels >= 0) # discard points walk out to the carmera scope
        update_pixels, edge_points, edge_uv = update_pixels[mask], edge_points[mask], edge_uv[mask]
        if mask.any():
            cnt = update_pixels.shape[0]
            update_pixels, unique_idx = unique(update_pixels, dim=0)
            unique_idx = torch.arange(cnt, device=update_pixels.device)[unique_idx]
            # assert update_pixels.shape == unique_idx.shape, f"{update_pixels.shape},{unique_idx.shape}"
            edge_points = edge_points[unique_idx]
            edge_uv = edge_uv[unique_idx]

            edge_mask.view(-1)[update_pixels] = True
        # edge_cnt = edge_mask.sum()
        # assert (
        #     edge_cnt == edge_points.shape[0]
        # ), f"{edge_cnt},{edge_points.shape},{edge_uv.shape},{update_pixels.shape},{torch.unique(update_pixels).shape},{update_pixels.min()},{update_pixels.max()}"
        # assert (
        #     edge_cnt == edge_uv.shape[0]
        # ), f"{edge_cnt},{edge_points.shape},{edge_uv.shape},{update_pixels.shape},{torch.unique(update_pixels).shape}"

    # ic(edge_mask.shape, edge_points.shape, edge_uv.shape)
    results = {"edge_mask": edge_mask, "edge_points": edge_points, "edge_uv": edge_uv, "edge_pixel_idx": update_pixels}

    if VERBOSE_MODE:  # debug
        edge_angles = torch.zeros_like(edge_mask).float()
        edge_sdf = torch.zeros_like(edge_mask).float().unsqueeze(-1)
        if edge_mask.any():
            ray_o = camera.get_camera_origin(prefix_shape=edge_points.shape[:-1])
            edge_viewdir = ray_o - edge_points
            edge_viewdir = edge_viewdir / (edge_viewdir.norm(dim=-1, keepdim=True) + 1e-10)
            with torch.enable_grad():
                edge_sdf_vals, _, edge_normals = sdf_network.get_all(edge_points, is_training=False)
            edge_normals = edge_normals / (edge_normals.norm(dim=-1, keepdim=True) + 1e-10)
            edge_dot = (edge_viewdir * edge_normals).sum(dim=-1)
            # edge_angles[edge_mask] = torch.rad2deg(torch.acos(edge_dot))
            # edge_sdf[edge_mask] = edge_sdf_vals
            edge_angles.view(-1)[update_pixels] = torch.rad2deg(torch.acos(edge_dot))
            edge_sdf.view(-1)[update_pixels] = edge_sdf_vals.squeeze(-1)

        results.update(
            {
                "walk_edge_found_mask": walk_edge_found_mask,
                "edge_angles": edge_angles,
                "edge_sdf": edge_sdf,
            }
        )

    return results


@torch.no_grad()
def raytrace_camera(
    camera,
    sdf_network,
    raytracer,
    max_num_rays=200000,
    fill_holes=False,
    detect_edges=False,
):
    results = raytrace_pixels(sdf_network, raytracer, camera.get_uv(), camera, max_num_rays=max_num_rays)
    results["depth"] *= results["convergent_mask"].float()

    if fill_holes:
        depth = results["depth"]
        kernel = torch.ones(3, 3).float().to(depth.device)
        depth = kornia.morphology.closing(depth.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)
        new_convergent_mask = depth > 1e-2
        update_mask = new_convergent_mask & (~results["convergent_mask"])
        if update_mask.any():
            results["depth"][update_mask] = depth[update_mask]
            results["convergent_mask"] = new_convergent_mask
            results["distance"] = results["depth"] * results["ray_d_norm"]
            results["points"] = results["ray_o"] + results["ray_d"] * results["distance"].unsqueeze(-1)

    if detect_edges:
        depth = results["depth"]
        convergent_mask = results["convergent_mask"]
        depth_grad_norm = kornia.filters.sobel(depth.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        depth_edge_mask = (depth_grad_norm > 1e-2) & convergent_mask # edge-walk points detection
        # depth_edge_mask = convergent_mask

        results.update(
            locate_edge_points(
                camera,
                results["points"],
                sdf_network,
                max_step=16,
                step_size=1e-3,
                dot_threshold=5e-2,
                max_num_rays=max_num_rays,
                mask=depth_edge_mask,
            )
        )
        results["convergent_mask"] &= ~results["edge_mask"]

        if VERBOSE_MODE:  # debug
            results.update({"depth_grad_norm": depth_grad_norm, "depth_edge_mask": depth_edge_mask})

    return results


def light_visibility(light_pos, points, sdf_network, inv_s, n_samples=128,  is_training=True):
    """
        light_pos: [3, ]
        points: [bs, 3]
        `light` and `points` should be in a training batch. 
    """
    device = points.device
    bs = points.shape[0]
    rays_d = (points - light_pos).to(device)
    norm_dir = torch.norm(rays_d, dim=-1, keepdim=True)
    rays_d = rays_d / norm_dir
    
    discard_samples = 10
    z_vals = torch.linspace(0, 1, n_samples, device=device)[:-discard_samples]
    n_samples = n_samples - discard_samples
    
    """
    integrate within the range of 127 points.
    """
    
    z_vals = z_vals.view(1, n_samples)
    z_vals = z_vals * norm_dir # [bs, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor(1./n_samples, device=device).expand(dists[..., :1].shape)], dim=-1)
    mid_z_vals = z_vals + dists * 0.5
    
    # pts, dirs: [bs, n_samples, 3]
    pts = light_pos + rays_d[..., None, :] * mid_z_vals[..., :, None]
    dirs = rays_d[..., None, :].expand(pts.shape)
    
    inv_s = inv_s.unsqueeze(1).expand(bs, n_samples, 1)
    pts = pts.reshape(-1, 3)
    dirs = dirs.reshape(-1, 3)
    inv_s = inv_s.reshape(-1, 1)
    # require gradiants
    sdf, _, gradients = sdf_network.get_all(pts, is_training=is_training)
    
    true_cos = (dirs * gradients).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos)) # simplified version
    
    estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf
    
    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(bs, n_samples).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([bs, 1], device=device).expand(bs, 1), 1.0 - alpha + 1e-7], -1), -1)[:, :-1]
    weights_sum = weights.sum(dim=-1, keepdim=True)
    acc_map = (1. - weights_sum).reshape(bs, ) # [bs, 1]
    ##### test ######
    # acc_map = 1. / (1. + ((acc_map-0.3) * (-20.)).exp())
    if not is_training:
        return acc_map.clone().detach()
    return acc_map


def render_light_visibility(
    results,
    camera,
    light_pos,
    sdf_network,
    inv_s,
    n_samples=128,
    max_num_pts=2500,
    is_training=True,
):
    dots_sh = list(results["convergent_mask"].shape)
    merge_acc_map = None
    # light_pos = camera.get_light_pos(offset)
    mask = (results["convergent_mask"] | results["edge_mask"])
    for points_split, mask_split in zip(
        torch.split(results["points"].view(-1, 3), max_num_pts, dim=0),
        torch.split(mask.view(-1), max_num_pts, dim=0),
    ):
        if mask_split.any():
            points_split = points_split[mask_split]
            acc_map_split = light_visibility(light_pos, points_split, sdf_network, inv_s, n_samples=n_samples, is_training=is_training)
        else:
            acc_map_split = (
                torch.Tensor([]).float().cuda(),
            )
        
        with torch.set_grad_enabled(is_training):
            acc_map_split_unmasked = torch.zeros(
                mask_split.shape,
                dtype=torch.float32,
                device=mask_split.device,
            )
            if mask_split.any():
                acc_map_split_unmasked[mask_split] = acc_map_split
            
            if merge_acc_map is None:
                merge_acc_map = [acc_map_split_unmasked]
            else:
                merge_acc_map.append(acc_map_split_unmasked)
    
    merge_acc_map = torch.cat(merge_acc_map, dim=0).reshape(
        dots_sh
    )
    results.update(
        {"vis": merge_acc_map,}
    )


def render_normal_and_color(
    results,
    sdf_network,
    color_network_dict,
    dino_network,
    render_fn,
    is_training=False,
    max_num_pts=320000,
    light_pos=None,
    roughness_warmup=False,
):
    """
    results: returned by raytrace_pixels function (updated after rendering)

    render interior and freespace pixels
    note: predicted color is black for freespace pixels
    """
    dots_sh = list(results["convergent_mask"].shape)

    merge_render_results = None
    for points_split, ray_d_split, ray_o_split, mask_split in zip(
        torch.split(results["points"].view(-1, 3), max_num_pts, dim=0),
        torch.split(results["ray_d"].view(-1, 3), max_num_pts, dim=0),
        torch.split(results["ray_o"].view(-1, 3), max_num_pts, dim=0),
        torch.split(results["convergent_mask"].view(-1), max_num_pts, dim=0),
    ):
        if mask_split.any():
            points_split, ray_d_split, ray_o_split = (
                points_split[mask_split],
                ray_d_split[mask_split],
                ray_o_split[mask_split],
            )
            sdf_split, feature_split, normal_split = sdf_network.get_all(points_split, is_training=is_training)
            dino_feature_split = dino_network(points_split, is_training=False)
            
            if is_training:
                points_split = reparam_points(points_split, normal_split.detach(), -ray_d_split.detach(), sdf_split)
        else:
            points_split, ray_d_split, ray_o_split, normal_split, feature_split, dino_feature_split = (
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
            )

        with torch.set_grad_enabled(is_training):
            lightdir = None
            distance = None
            if light_pos is not None:
                if mask_split.any():
                    lightdir = F.normalize(points_split - light_pos, eps=1e-10)
                    distance = (points_split - light_pos).norm(dim=-1, keepdim=True)
                else:
                    lightdir = torch.Tensor([]).float().cuda()
            
            render_results = render_fn(
                mask_split,
                color_network_dict,
                ray_o_split,
                ray_d_split,
                points_split,
                normal_split,
                feature_split,
                dino_f=dino_feature_split,
                is_training=is_training,
                lightdir=lightdir,
                distance=distance,
                roughness_warmup=roughness_warmup,
            )

            if merge_render_results is None:
                merge_render_results = dict([(x, [render_results[x],]) for x in render_results.keys()])
            else:
                for x in render_results.keys():
                    merge_render_results[x].append(render_results[x])
                    

    for x in list(merge_render_results.keys()):
        tmp = torch.cat(merge_render_results[x], dim=0).reshape(dots_sh + [-1,])
        if tmp.shape[-1] == 1:
            tmp = tmp.squeeze(-1)
        merge_render_results[x] = tmp

    results.update(merge_render_results)
    
    # create indices map for visualization
    # convergent_mask = results["convergent_mask"]
    # print(f"convergent_mask.shape {convergent_mask.shape}")


def render_edge_pixels(
    results,
    camera,
    sdf_network,
    raytracer,
    color_network_dict,
    dino_network,
    render_fn,
    is_training=False,
    light_pos=None,
    roughness_warmup=False,
):
    edge_mask, edge_points, edge_uv, edge_pixel_idx = (
        results["edge_mask"],
        results["edge_points"],
        results["edge_uv"],
        results["edge_pixel_idx"],
    )
    edge_pixel_center = torch.floor(edge_uv) + 0.5

    edge_sdf, _, edge_grads = sdf_network.get_all(edge_points, is_training=is_training)
    edge_normals = edge_grads.detach() / (edge_grads.detach().norm(dim=-1, keepdim=True) + 1e-10)
    if is_training:
        edge_points = reparam_points(edge_points, edge_grads.detach(), edge_normals, edge_sdf)
        edge_uv = camera.project(edge_points)

    edge_normals2d = torch.matmul(edge_normals, camera.W2C[:3, :3].transpose(1, 0))[:, :2] # discard the last term `1`
    edge_normals2d = edge_normals2d / (edge_normals2d.norm(dim=-1, keepdim=True) + 1e-10)

    # sample a point on both sides of the edge
    # approximately think of each pixel as being approximately a circle with radius 0.707=sqrt(2)/2
    # see paper https://arxiv.org/pdf/2204.02232 Eq. 7, 8, 9
    pixel_radius = 0.707
    pos_side_uv = edge_pixel_center - pixel_radius * edge_normals2d
    neg_side_uv = edge_pixel_center + pixel_radius * edge_normals2d

    dot2d = torch.sum((edge_uv - edge_pixel_center) * edge_normals2d, dim=-1)
    alpha = 2 * torch.arccos(torch.clamp(dot2d / pixel_radius, min=0.0, max=1.0))
    pos_side_weight = 1.0 - (alpha - torch.sin(alpha)) / (2.0 * np.pi)

    # render positive-side and negative-side colors by raytracing; speed up using edge mask
    pos_side_results = raytrace_pixels(sdf_network, raytracer, pos_side_uv, camera)
    neg_side_results = raytrace_pixels(sdf_network, raytracer, neg_side_uv, camera)
    render_normal_and_color(pos_side_results, sdf_network, color_network_dict, dino_network, render_fn,
                            is_training=is_training, light_pos=light_pos, roughness_warmup=roughness_warmup)
    render_normal_and_color(neg_side_results, sdf_network, color_network_dict, dino_network, render_fn,
                            is_training=is_training, light_pos=light_pos, roughness_warmup=roughness_warmup)
    # assign colors to edge pixels
    edge_color = pos_side_results["color"] * pos_side_weight.unsqueeze(-1) \
                + neg_side_results["color"] * (1.0 - pos_side_weight.unsqueeze(-1))

    results["color"].view(-1, 3)[edge_pixel_idx] = edge_color
    results["normal"].view(-1, 3)[edge_pixel_idx] = edge_grads
    
    # The following code is just for visualize, you can comment out to speed up
    # Material Visualization
    results["diffuse_albedo"].view(-1, 3)[edge_pixel_idx] = \
        pos_side_results["diffuse_albedo"] * pos_side_weight.unsqueeze(-1) + \
        neg_side_results["diffuse_albedo"] * (1.0 - pos_side_weight.unsqueeze(-1))
    results["specular_albedo"].view(-1, 3)[edge_pixel_idx] = \
        pos_side_results["specular_albedo"] * pos_side_weight.unsqueeze(-1) + \
        neg_side_results["specular_albedo"] * (1.0 - pos_side_weight.unsqueeze(-1))
    results["specular_roughness"].view(-1, )[edge_pixel_idx] = \
        pos_side_results["specular_roughness"] * pos_side_weight + \
        neg_side_results["specular_roughness"] * (1.0 - pos_side_weight)
    # DINO feature Visualization
    results["dino_feature"].view(-1, dimension)[edge_pixel_idx] = \
        pos_side_results["dino_feature"] * pos_side_weight.unsqueeze(-1) + \
        neg_side_results["dino_feature"] * (1.0 - pos_side_weight.unsqueeze(-1))
    
    results["edge_pos_neg_normal"] = torch.cat(
        [
            pos_side_results["normal"][pos_side_results["convergent_mask"]],
            neg_side_results["normal"][neg_side_results["convergent_mask"]],
        ],
        dim=0,
    )

    results["uv"].view(-1, 2)[edge_pixel_idx] = edge_uv.detach()
    results["points"].view(-1, 3)[edge_pixel_idx] = edge_points.detach()

    if VERBOSE_MODE:
        pos_side_weight_fullsize = torch.zeros_like(edge_mask).float()
        # pos_side_weight_fullsize[edge_mask] = pos_side_weight
        pos_side_weight_fullsize.view(-1)[edge_pixel_idx] = pos_side_weight

        pos_side_depth = torch.zeros_like(edge_mask).float()
        # pos_side_depth[edge_mask] = pos_side_results["depth"]
        pos_side_depth.view(-1)[edge_pixel_idx] = pos_side_results["depth"]
        neg_side_depth = torch.zeros_like(edge_mask).float()
        # neg_side_depth[edge_mask] = neg_side_results["depth"]
        neg_side_depth.view(-1)[edge_pixel_idx] = neg_side_results["depth"]

        pos_side_color = (
            torch.zeros(
                list(edge_mask.shape)
                + [
                    3,
                ]
            )
            .float()
            .to(edge_mask.device)
        )
        # pos_side_color[edge_mask] = pos_side_results["color"]
        pos_side_color.view(-1, 3)[edge_pixel_idx] = pos_side_results["color"]
        neg_side_color = (
            torch.zeros(
                list(edge_mask.shape)
                + [
                    3,
                ]
            )
            .float()
            .to(edge_mask.device)
        )
        # neg_side_color[edge_mask] = neg_side_results["color"]
        neg_side_color.view(-1, 3)[edge_pixel_idx] = neg_side_results["color"]
        results.update(
            {
                "edge_pos_side_weight": pos_side_weight_fullsize,
                "edge_normals2d": edge_normals2d,
                "pos_side_uv": pos_side_uv,
                "neg_side_uv": neg_side_uv,
                "edge_pos_side_depth": pos_side_depth,
                "edge_neg_side_depth": neg_side_depth,
                "edge_pos_side_color": pos_side_color,
                "edge_neg_side_color": neg_side_color,
            }
        )


def render_indirect_lighting(
    results,
    sdf_network,
    color_network_dict,
    dino_network,
    raytracer,
    render_fn,
    light_pos=None,
    roughness_warmup=False,
    is_training=False,
    sec_is_training=False,
    diff_geo=True,
):
    points = results["points"]
    ray_o = results["ray_o"]
    ray_d = results["ray_d"]
    points_mask = results["convergent_mask"]
    dots_sh = list(points.shape[:-1])
    device = points.device
    W, H, D = points.shape
    
    # mask the points that need to be handled
    points = points[points_mask].clone().reshape(-1, 3) # [hit_points, 3]
    ray_o = ray_o[points_mask].clone().reshape(-1, 3)   # [hit_points, 3]
    ray_d = ray_d[points_mask].clone().reshape(-1, 3)   # [hit_points, 3]
    merged_results = None
    indirect_color = torch.zeros(dots_sh + [3, ], device=device)
    indirect_blend_coef = torch.ones(dots_sh, dtype=torch.float32, device=device)
    if points_mask.any():
        for points, ray_o, ray_d in zip(
            torch.split(points, 200000, dim=0),
            torch.split(ray_o, 200000, dim=0),
            torch.split(ray_d, 200000, dim=0),
        ):
            normals = sdf_network.gradient(points).clone().detach()
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
            with torch.no_grad():
                # sample_dirs = uniform_random_hemisphere(normals)
                sample_dirs = torch.sum(-ray_d * normals, dim=-1, keepdim=True) * normals * 2 + ray_d
                sample_dirs = sample_dirs / (sample_dirs.norm(dim=-1, keepdim=True) + 1e-10)
                # pdf = torch.ones_like(sample_dirs[:, 0:1], device=device) * torch.pi
        
            sec_object_mask = torch.ones(points.shape[0], device=device).bool()
            sample_dirs = sample_dirs.reshape(-1, 3)
            # find second intersection
            sdf = lambda x: sdf_network(x)[:, 0]
            with torch.no_grad():
                mask_intersect_split, min_dis_split, max_dis_split = intersect_sphere(points, sample_dirs, r=1.0)
                ret_dict = raytracer(
                    sdf,
                    points, # ray_o: [N, 3]
                    sample_dirs, # ray_d: [N, 3]
                    min_dis_split.clamp(min=0.001, max=3.00),
                    max_dis_split.clamp(min=0.001, max=3.00),
                    mask_intersect_split & sec_object_mask, # [N, ]
                )
                sec_net_object_mask, sec_points = ret_dict["convergent_mask"], ret_dict["points"]
            # check light visibility of secondary points (uncolocated setting)
            if light_pos is not None:
                z_vals = torch.linspace(0.01, 0.8, 20, device=device)
                mask_sdf = torch.zeros_like(sec_net_object_mask, device=device).bool()
                mask_sdf[sec_net_object_mask] = True
                dirs = ray_o[sec_net_object_mask] - sec_points[sec_net_object_mask]
                dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-10)
                # ic(sec_points[sec_net_object_mask].shape, dirs.shape, z_vals.shape)
                pts = sec_points[sec_net_object_mask][:, None, :] + dirs[:, None, :] * z_vals[None, :, None] # [cnt, 20, 3]
                pts_sdf = sdf(pts.view(-1, 3)).view(pts.shape[0], 20)
                # sec_points -> light, is occluded
                mask_sdf[sec_net_object_mask] = ~((pts_sdf < 0).sum(dim=-1) > 0)
                sec_net_object_mask = mask_sdf
            
            # * secondary points rendering
            hit_points = sec_points[sec_net_object_mask]
            hit_ray_o = points[sec_net_object_mask]
            hit_ray_d = sample_dirs[sec_net_object_mask]
            if light_pos is None: # co-located setting
                lightdir = hit_points - ray_o[sec_net_object_mask]
            else:
                lightdir = hit_points - light_pos
            hit_distance = lightdir.norm(dim=-1, keepdim=True)
            lightdir = lightdir / (lightdir.norm(dim=-1, keepdim=True) + 1e-10)
            # ? reparam points
            hit_sdf, hit_feature, hit_normal = sdf_network.get_all(hit_points, is_training=sec_is_training)
            hit_dino_feature = dino_network(hit_points, is_training=sec_is_training)
            if diff_geo and sec_is_training:
                hit_points = reparam_points(hit_points, hit_normal.detach(), -hit_ray_d.detach(), hit_sdf)
            # if use_VQ:
            #     hit_clip_feature, _, _ = clip_network(hit_points, hit_normal, hit_feature, is_training=sec_is_training, use_VQ=True)
            #     hit_rough_feature, _, _ = rough_network(hit_points, hit_normal, hit_feature, is_training=sec_is_training, use_VQ=True)
            # else:
            #     hit_clip_feature = clip_network(hit_points, hit_normal, hit_feature, is_training=sec_is_training, use_VQ=False)
            #     hit_rough_feature = rough_network(hit_points, hit_normal, hit_feature, is_training=sec_is_training, use_VQ=False)
            
            with torch.set_grad_enabled(sec_is_training):
                indirect_results = render_fn(
                    sec_net_object_mask,
                    color_network_dict,
                    hit_ray_o,
                    hit_ray_d,
                    hit_points,
                    hit_normal,
                    hit_feature,
                    dino_f=hit_dino_feature,
                    light=None,
                    lightdir=lightdir,
                    distance=hit_distance,
                    roughness_warmup=roughness_warmup,
                    is_training=sec_is_training,
                    EDirectRadiance=True,
                ) # lightdir -> hit_points -> (-hit_ray_d)
            
            # trace_radiance, sample_dirs, results["ray_o"], results["ray_d"], results["points"] (lightdir = ray_d)
            # synthesize indirect lighting RGB
            ray_o = ray_o[sec_net_object_mask]
            ray_d = ray_d[sec_net_object_mask]
            points = points[sec_net_object_mask]
            sample_dirs = sample_dirs[sec_net_object_mask]
            trace_radiance = indirect_results["color"][sec_net_object_mask]
            
            points_sdf, points_feature, points_normal = sdf_network.get_all(points, is_training=is_training)
            dino_points_feature = dino_network(points, is_training=False)
            # ? reparam points
            if diff_geo and is_training:
                points = reparam_points(points, points_normal.detach(), -ray_d.detach(), points_sdf)
            
            with torch.set_grad_enabled(is_training):
                rendered_indirect_results = render_fn(
                    sec_net_object_mask,
                    color_network_dict,
                    ray_o,
                    ray_d, 
                    points,
                    points_normal,
                    points_feature,
                    dino_points_feature,
                    light=trace_radiance, # 2 * pi
                    lightdir=-sample_dirs,
                    distance=None,
                    roughness_warmup=roughness_warmup,
                    is_training=is_training,
                    EDirectRadiance=False,
                )
                blend_coef = torch.ones(list(sec_net_object_mask.shape), dtype=torch.float32, device=sec_net_object_mask.device)
                if sec_net_object_mask.any():
                    # FIXME: detach or not ?
                    roughness = rendered_indirect_results["specular_roughness"][sec_net_object_mask].reshape(-1, 1).detach()
                    normals = points_normal.detach()
                    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
                    dot_n_wo = torch.sum(sample_dirs * normals, dim=-1, keepdim=True) # (-1, 1)
                    distance = torch.norm(points - hit_points, dim=-1, keepdim=True)
                    blend_coef[sec_net_object_mask] = color_network_dict["blend_network"](distance, roughness, dot_n_wo).squeeze(-1)
            
            if merged_results is None:
                merged_results = {
                    "indirect_color": [rendered_indirect_results["color"]],
                    "indirect_blend_coef": [blend_coef],
                    # "indirect_diffuse_color": [rendered_indirect_results["diffuse_color"]],
                    # "indirect_specular_color": [rendered_indirect_results["specular_color"]],
                }
            else:
                merged_results["indirect_color"].append(rendered_indirect_results["color"])
                merged_results["indirect_blend_coef"].append(blend_coef)
                # merged_results["indirect_diffuse_color"].append(rendered_indirect_results["diffuse_color"])
                # merged_results["indirect_specular_color"].append(rendered_indirect_results["specular_color"])
        
        for x in merged_results.keys():
            merged_results[x] = torch.cat(merged_results[x], dim=0)
        
        indirect_color[points_mask] = merged_results["indirect_color"]
        indirect_blend_coef[points_mask] = merged_results["indirect_blend_coef"]
        # indirect_diffuse_color[points_mask] = merged_results["indirect_diffuse_color"]
        # indirect_specular_color[points_mask] = merged_results["indirect_specular_color"]
        # results.update(
        #     {
        #         "indirect_color": indirect_color,
        #         "indirect_diffuse_color": indirect_diffuse_color,
        #         "indirect_specular_color": indirect_specular_color,
        #     }
        # )
    
    # * merge pixels
    results.update(
        {
            "indirect_color": indirect_color.reshape(W, H, 3),
            "indirect_blend_coef": indirect_blend_coef.reshape(W, H, 1),
        }
    )
    # results["color"] = results["color"].reshape(W, H, 3)
    # results["diffuse_color"] = results["diffuse_color"].reshape(W, H, 3)
    # results["specular_color"] = results["specular_color"].reshape(W, H, 3)
    # results["convergent_mask"] = (results["convergent_mask"].sum(dim=-1) > 0)
    
    # results["color"] = results["color"] + results["indirect_color"] * results["indirect_blend_coef"]


def render_camera(
    camera,
    sdf_network,
    raytracer,
    color_network_dict,
    dino_network,
    render_fn,
    fill_holes=True,
    handle_edges=True,
    inv_s=0.0,
    render_visibility=False,
    max_num_pts=512,
    light_pos=None,
    roughness_warmup=False,
    indirect_warmup=False,
    is_training=False,
):
    results = raytrace_camera(
        camera,
        sdf_network,
        raytracer,
        max_num_rays=200000,
        fill_holes=fill_holes,
        detect_edges=handle_edges,
    )
    
    if render_visibility:
        render_light_visibility(
            results,
            camera,
            light_pos,
            sdf_network,
            inv_s,
            n_samples=128,
            max_num_pts=max_num_pts,
            is_training=is_training,
        )

    render_normal_and_color(
        results,
        sdf_network,
        color_network_dict,
        dino_network,
        render_fn,
        is_training=is_training,
        max_num_pts=320000,
        light_pos=light_pos,
        roughness_warmup=roughness_warmup,
    )
    
    if not indirect_warmup:
        render_indirect_lighting(
            results,
            sdf_network,
            color_network_dict,
            dino_network,
            raytracer,
            render_fn,
            roughness_warmup=roughness_warmup,
            is_training=is_training,
            diff_geo=True,
        )
    
    if handle_edges and results["edge_mask"].sum() > 0:
        render_edge_pixels(
            results,
            camera,
            sdf_network,
            raytracer,
            color_network_dict,
            dino_network,
            render_fn,
            is_training=is_training,
            light_pos=light_pos,
            roughness_warmup=roughness_warmup,
        )
    
    return results


def render_camera_visibility(
    camera,
    sdf_network,
    raytracer,
    fill_holes=True,
    handle_edges=True,
    is_training=False,
    inv_s=0.0,
    render_visibility=False,
    max_num_pts=4000,
    light_pos=None,
):
    results = raytrace_camera(
        camera,
        sdf_network,
        raytracer,
        max_num_rays=200000,
        fill_holes=fill_holes,
        detect_edges=handle_edges,
    )
    
    if render_visibility:
        render_light_visibility(
            results,
            camera,
            light_pos,
            sdf_network,
            inv_s,
            n_samples=256,
            max_num_pts=max_num_pts,
            is_training=is_training,
        )
    return results


def render_dino(
    results,
    dino_network,
    max_num_pts=320000,
    is_training=True,
):
    dots_sh = list(results["convergent_mask"].shape)
    merge_dino_results = None

    for points_split, mask_split in zip(
        torch.split(results["points"].view(-1, 3), max_num_pts, dim=0),
        torch.split(results["convergent_mask"].view(-1), max_num_pts, dim=0),
    ):
        if mask_split.any():
            points_split = points_split[mask_split]
            dino_feature_split_masked = dino_network(points_split, is_training=is_training)
        else:
            points_split, dino_feature_split_masked = (
                torch.Tensor([]).float().cuda(),
                torch.Tensor([]).float().cuda(),
            )
            
        with torch.enable_grad():
            dino_feature_split = torch.zeros(
                list(mask_split.shape) + [dimension,],
                dtype=torch.float32,
                device=mask_split.device,
            )
            if mask_split.any():
                dino_feature_split[mask_split] = dino_feature_split_masked
            
            if merge_dino_results is None:
                merge_dino_results = [dino_feature_split]
            else:
                merge_dino_results.append(dino_feature_split)
    
    merge_dino_results = torch.cat(merge_dino_results, dim=0).reshape(
        dots_sh + [-1,]
    )
    results.update({"dino": merge_dino_results})


def dino_camera(
    camera,
    sdf_network,
    raytracer,
    dino_network,
    is_training=True,
):
    """
    raytrace the camera, and render the surface dino feature
    """
    results = raytrace_camera(
        camera,
        sdf_network,
        raytracer,
        max_num_rays=200000,
    )
    
    render_dino(
        results,
        dino_network,
        max_num_pts=320000,
        is_training=is_training,
    )
    
    return results # loss(results["dino"], dino_gt)

