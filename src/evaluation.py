import numpy as np
import imageio.v2 as imageio
import os
from skimage.metrics import structural_similarity
import lpips
import torch
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib import cm
import math
from tqdm import tqdm
import argparse
import time

def skimage_ssim(pre_im, gt_im):
    ssim = 0.
    if pre_im.shape[-1] == 1 or gt_im.shape[-1] == 1:
        ssim = structural_similarity(gt_im[:, :, 0], pre_im[:, :, 0], 
                                    data_range=1.0, win_size=11, sigma=1.5,
                                    use_sample_covariance=False, k1=0.01, k2=0.03)
    else:
        for ch in range(3):
            ssim += structural_similarity(gt_im[:, :, ch], pre_im[:, :, ch], 
                                        data_range=1.0, win_size=11, sigma=1.5,
                                        use_sample_covariance=False, k1=0.01, k2=0.03)
        ssim /= 3.
    return ssim

def read_image(fpath):
    img = imageio.imread(fpath).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    if not fpath.endswith('.exr'):
        img = img / 255.
    # img = np.clip(img, 0., 1.)
    return img

mse2psnr = lambda x: -10. * np.log(x+1e-10) / np.log(10.)

def PSNR(img1, img2, mask=None):
    '''
    Input : H x W x 3   [0,1]
    Output : PSNR
    '''
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    if mask is not None:
        img1, img2 = img1[mask.astype(bool)], img2[mask.astype(bool)]
    mse = np.mean((img1 - img2)**2)
    psnr = - 10.0 * np.log10(mse)
    return psnr

def calculate_mse(img1, img2, mask=None):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    mse = np.mean((img1 - img2) ** 2)
    return mse

def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [0, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid  = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte() > 0]
    n_err_mean = ang_valid.sum() / valid
    n_err_med  = ang_valid.median()
    n_acc_11   = (ang_valid < 11.25).sum().float() / valid
    n_acc_30   = (ang_valid < 30).sum().float() / valid
    n_acc_45   = (ang_valid < 45).sum().float() / valid
    
    def colorMap(diff):
        thres = 90
        diff_norm = np.clip(diff, 0, thres) / thres
        diff_cm = torch.from_numpy(cm.jet(diff_norm.numpy()))[:,:,:, :3]
        return diff_cm.permute(0,3,1,2).clone().float()

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), 
            'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()}
    angular_error_map = {'angular_map': angular_map}
    return value, angular_error_map


def align_(rgb_gt, rgb_pre, mask, eps=1e-4):
    for c in range(rgb_gt.shape[2]):
        gt_value = rgb_gt[..., c:c+1][mask]
        pre_value = rgb_pre[..., c:c+1][mask]
        pre_value[pre_value<=eps]=eps
        scale = np.median(gt_value / pre_value)
        # scale = np.mean(gt_value) / np.mean(pre_value)
        rgb_pre[..., c] *= scale
        

def scale_(rgb, scale):
    for c in range(rgb.shape[2]):
        rgb[..., c] *= scale
        rgb[..., c] = np.clip(rgb[..., c], 0., 1.)


def read_light(file_name):
    with open(file_name, 'r') as file:
        return float(file.readline().strip())

def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = np.float32(alpha) / 255.
    object_mask = alpha > 0.5

    return object_mask

def evaluation(source_dir, gt_dir, item='rgb', use_mask=False, align=False, tonemap=False, plot_diff_map=True):
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_mse = []
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).cuda()
    
    gt_dir = os.path.join(gt_dir, 'image' if item == 'rgb' else item) # eval folder
    
    tonemap_img = lambda x: np.power(x, 1. / 2.2)
    clip_img = lambda x: np.clip(x, 0., 1.)
    
    if 'albedo' in item and not align:
        pre_light = read_light(os.path.join(source_dir, 'light.txt'))
        gt_light = read_light(os.path.join(gt_dir, '../../light.txt'))
        scale = pre_light / gt_light
        print(f'[{item}]: pre_light: {pre_light}, gt_light: {gt_light}, scale: {scale}')
    else:
        scale = 1.0
    
    if align:
        metric_path = os.path.join(source_dir, f'./metrics_{item}_align.txt')
    else:
        metric_path = os.path.join(source_dir, f'./metrics_{item}.txt')
    # Enter source_dir
    # source_dir = os.path.join(source_dir, item)
    with open(metric_path, 'w') as fp:
        fp.write('img_name\tpsnr\tssim\tlpips\n')
        if item == 'rgb':
            img_list = sorted(glob.glob(os.path.join(source_dir, item, '*.png')), key=lambda x : int(x.split('/')[-1].split('.')[0]))
        else:
            img_list = sorted(glob.glob(os.path.join(source_dir, item, '*.exr')), key=lambda x : int(x.split('/')[-1].split('.')[0]))
        
        for fpath in tqdm(img_list):
            name = os.path.basename(fpath)
            idx = int(name[:-4])
            suffix = name[-4:]
            if suffix == '.png':
                pre_im = read_image(os.path.join(source_dir, item, '{}.png'.format(idx)))
                gt_im = read_image(os.path.join(gt_dir, '{}.png'.format(idx)))
                scale_(pre_im, scale)
            elif suffix == '.exr':
                pre_im = read_image(os.path.join(source_dir, item, '{}.exr'.format(idx)))
                gt_im = read_image(os.path.join(gt_dir, '{}.exr'.format(idx)))
                scale_(pre_im, scale)
            
            h, w, c = gt_im.shape
            valid = h*w
            mask = None
            if use_mask:
                mask = load_mask(os.path.join(source_dir, 'mask', '{}_mask.png'.format(idx)))
                # FIXME: need GT mask
                pre_im[:, :, 0:1][~mask] = 0
                pre_im[:, :, 1:2][~mask] = 0
                pre_im[:, :, 2:][~mask] = 0
                gt_im[:, :, 0:1][~mask] = 0
                gt_im[:, :, 1:2][~mask] = 0
                gt_im[:, :, 2:][~mask] = 0
                
            if tonemap:
                pre_im = clip_img(tonemap_img(pre_im))
                gt_im = clip_img(tonemap_img(gt_im))
            if align:
                assert use_mask
                align_(pre_im, gt_im, mask)
                pre_im = clip_img(pre_im)
                gt_im = clip_img(gt_im)
            # plot diff map
            if plot_diff_map:
                diff_map = np.abs(pre_im - gt_im).mean(2)
                diff_map = np.clip(diff_map, 0.0, 0.50) / 0.50 # threshold = 0.5
                diff_map_color = cm.jet(diff_map)  # Normalize to range [0, 1] for color mapping
                os.makedirs(os.path.join(source_dir, 'diff'), exist_ok=True)
                plt.imsave(os.path.join(os.path.join(source_dir, 'diff'), '{}_diff.png'.format(idx)), diff_map_color)
            # PSNR
            psnr = PSNR(pre_im, gt_im)
            # SSIM
            ssim = skimage_ssim(gt_im, pre_im)
            # LPIPS
            pre_im = torch.from_numpy(pre_im).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
            gt_im = torch.from_numpy(gt_im).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
            d = loss_fn_alex(gt_im.cuda(), pre_im.cuda()).item()
            # angular error
            # normal_pred = torch.from_numpy(np.load(os.path.join(source_dir, 'normal', 'img_{}_normal.npy'.format(idx)))).permute(2, 0, 1).unsqueeze(0)
            # normal_gt = torch.from_numpy(np.load(os.path.join(gt_dir, 'img_{}_normal.npy'.format(idx)))).permute(2, 0, 1).unsqueeze(0)
            # mask = torch.from_numpy(mask[np.newaxis, np.newaxis, ...])
            # value, angular_error_map = calNormalAcc(normal_gt, normal_pred, mask)
            # mae = value['n_err_mean']
            fp.write('{}.png\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(idx, psnr, ssim, d))

            all_psnr.append(psnr)
            all_ssim.append(ssim)
            all_lpips.append(d)
        fp.write('\nAverage\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips)))


def eval_all(source_dir, gt_dir):
    print(f"Start evaluation...")
    print(f"source_dir: {source_dir}")
    # eval RGB
    evaluation(source_dir, gt_dir, use_mask=False, align=False, tonemap=False)
    # eval diffuse_albedo
    evaluation(source_dir, gt_dir,
               item='diffuse_albedo',
               use_mask=True,
               align=False,
               tonemap=False,
               plot_diff_map=False)
    # eval aligned diffuse_albedo
    evaluation(source_dir, gt_dir,
               item='diffuse_albedo',
               use_mask=True,
               align=True,
               tonemap=False,
               plot_diff_map=False)
    # eval specular_albedo
    evaluation(source_dir, gt_dir,
               item='specular_albedo',
               use_mask=True,
               align=False,
               tonemap=False,
               plot_diff_map=False)
    # eval aligned specular_albedo
    evaluation(source_dir, gt_dir,
               item='specular_albedo',
               use_mask=True,
               align=True,
               tonemap=False,
               plot_diff_map=False)
    # eval roughness
    eval_roughness(source_dir, gt_dir, use_mask=False)
    print("Done.")


def eval_roughness(source_dir, gt_dir, use_mask=False):
    all_mse = []
    gt_dir = os.path.join(gt_dir, 'specular_roughness') # evaluation on the image folder    
    metric_path = os.path.join(source_dir, './metrics_roughness.txt')
    with open(metric_path, 'w') as fp:
        fp.write('img_name\tmse\n')
        img_list = sorted(glob.glob(os.path.join(source_dir, 'specular_roughness', '*.exr')), key=lambda x : int(x.split('/')[-1].split('.')[0]))
        for fpath in tqdm(img_list):
            name = os.path.basename(fpath)
            idx = int(name[:-4])
            pre_im = read_image(os.path.join(source_dir, 'specular_roughness', '{}.exr'.format(idx)))
            gt_im = read_image(os.path.join(gt_dir, '{}.exr'.format(idx)))
            
            h, w, c = gt_im.shape
            valid = h*w
            mask = None
            if use_mask:
                mask = imageio.imread(os.path.join(source_dir, 'mask', '{}_mask.png'.format(idx))).astype(bool)

            mse = calculate_mse(pre_im, gt_im, mask)
            fp.write('{}.exr\t{:.6f}\n'.format(idx, mse))
            all_mse.append(mse)
        fp.write('\nAverage\t{:.6f}\n'.format(np.mean(all_mse)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python evaluation.py --source_dir ../exp_iron_stage2/drv/tree/render_test_60000 --gt_dir /data2/jingzhi/IRON/data_flashlight/drv/tree/test
    parser.add_argument('--source_dir', type=str,default='', help='path to rendering folder')
    parser.add_argument('--gt_dir', type=str, default='', help='path to ground truth')
    args = parser.parse_args()
    
    print(f"source_dir: {args.source_dir}")
    print(f"gt_dir: {args.gt_dir}")
    
    print('Start evaluation...')
    start_time = time.time()
    eval_all(args.source_dir, args.gt_dir)
    end_time = time.time()
    print(f'Eval done in {end_time - start_time:.2f}s')
