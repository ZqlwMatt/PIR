import torch
import torch.nn as nn
import numpy as np
import os


### https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L477
def smithG1(cosTheta, alpha):
    sinTheta = torch.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + torch.hypot(root, torch.ones_like(root)))

def G_fn(dot_wi_n, dot_wo_n, alpha):
    return smithG1(dot_wi_n, alpha) * smithG1(dot_wo_n, alpha)

class GGXColocatedRenderer(nn.Module):
    """Usage: forward(self, light, distance, normal, viewdir, diffuse_albedo, specular_albedo, alpha)"""
    def __init__(self, use_cuda=False):
        super().__init__()

        self.MTS_TRANS = torch.from_numpy(
            np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/ext_mts_rtrans_data.txt")).astype(
                np.float32
            )
        )  # 5000 entries, external IOR
        self.MTS_DIFF_TRANS = torch.from_numpy(
            np.loadtxt(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggx/int_mts_diff_rtrans_data.txt")
            ).astype(np.float32)
        )  # 50 entries, internal IOR
        self.num_theta_samples = 100
        self.num_alpha_samples = 50

        if use_cuda:
            self.MTS_TRANS = self.MTS_TRANS.cuda()
            self.MTS_DIFF_TRANS = self.MTS_DIFF_TRANS.cuda()

    def forward(self, light, distance, normal, lightdir, viewdir, diffuse_albedo, specular_albedo, alpha, EDirectRadiance=True):
        """
        light:
        distance: [..., 1]
        normal, viewdir: [..., 3]; both normal and viewdir point away from objects
        diffuse_albedo, specular_albedo: [..., 3]
        alpha: [..., 1]; roughness
        """
        # decay light according to squared-distance falloff
        if EDirectRadiance:
            # distance = torch.clamp(distance, min=1.0)
            light_intensity = light / (distance * distance + 1e-10)
        else:
            light_intensity = light

        # <wo, n> = <w_i, n> = <h, n> in colocated setting
        dot_wo_n = torch.sum(viewdir * normal, dim=-1, keepdims=True)
        dot_wi_n = torch.sum(lightdir * normal, dim=-1, keepdims=True)
        dot_wo_n = torch.clamp(dot_wo_n, min=0.00001, max=0.99999)  # must be very precise; cannot be 0.999
        dot_wi_n = torch.clamp(dot_wi_n, min=0.00001, max=0.99999)
        
        H = viewdir + lightdir # [N, 3]
        H = nn.functional.normalize(H, dim=-1, eps=1e-10)
        dot_h_n = torch.sum(H * normal, dim=-1, keepdims=True)
        dot_h_n = torch.clamp(dot_h_n, min=0.00001, max=0.99999)
        # default value of IOR['polypropylene'] / IOR['air'].
        m_eta = 1.48958738
        m_invEta2 = 1.0 / (m_eta * m_eta)

        # clamp alpha for numeric stability
        alpha = torch.clamp(alpha, min=0.0001)

        # specular term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L347
        ## compute GGX NDF: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L191
        cosTheta2_H = dot_h_n * dot_h_n
        root = cosTheta2_H + (1.0 - cosTheta2_H) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        ## compute fresnel: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/libcore/util.cpp#L651
        # F = 0.04
        F = 0.03867

        ## compute shadowing term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/microfacet.h#L520
        G = G_fn(dot_wi_n, dot_wo_n, alpha)  # [..., 1]

        specular_rgb = light_intensity * specular_albedo * F * D * G / (4.0 * dot_wi_n + 1e-10)

        # diffuse term: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/roughplastic.cpp#L367
        ## compute T12: : https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L183
        ### data_file: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L93
        ### assume eta is fixed
        warpedCosTheta_wi = dot_wi_n**0.25
        warpedCosTheta_wo = dot_wo_n**0.25
        alphaMin, alphaMax = 0, 4
        warpedAlpha = ((alpha - alphaMin) / (alphaMax - alphaMin)) ** 0.25  # [..., 1]
        
        tx_wi = torch.floor(warpedCosTheta_wi * self.num_theta_samples).long()
        ty_wi = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        t_idx_wi = ty_wi * self.num_theta_samples + tx_wi
        tx_wo = torch.floor(warpedCosTheta_wo * self.num_theta_samples).long()
        ty_wo = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        t_idx_wo = ty_wo * self.num_theta_samples + tx_wo

        dots_sh = list(t_idx_wi.shape[:-1])
        data = self.MTS_TRANS.view([1,] * len(dots_sh) + [-1,]).expand(
            dots_sh
            + [
                -1,
            ]
        )

        t_idx_wi = torch.clamp(t_idx_wi, min=0, max=data.shape[-1] - 1).long()  # important
        t_idx_wo = torch.clamp(t_idx_wo, min=0, max=data.shape[-1] - 1).long()  # important
        T12 = torch.clamp(torch.gather(input=data, index=t_idx_wi, dim=-1), min=0.0, max=1.0)
        T21 = torch.clamp(torch.gather(input=data, index=t_idx_wo, dim=-1), min=0.0, max=1.0)

        ## compute Fdr: https://github.com/mitsuba-renderer/mitsuba/blob/cfeb7766e7a1513492451f35dc65b86409655a7b/src/bsdfs/rtrans.h#L249
        t_idx = torch.floor(warpedAlpha * self.num_alpha_samples).long()
        data = self.MTS_DIFF_TRANS.view([1,] * len(dots_sh) + [-1,]).expand(
            dots_sh
            + [
                -1,
            ]
        )
        t_idx = torch.clamp(t_idx, min=0, max=data.shape[-1] - 1).long()  # important
        Fdr = torch.clamp(1.0 - torch.gather(input=data, index=t_idx, dim=-1), min=0.0, max=1.0)  # [..., 1]

        diffuse_rgb = light_intensity * (diffuse_albedo / (1.0 - Fdr + 1e-10) / np.pi) * dot_wo_n * T12 * T21 * m_invEta2
        ret = {"diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb, "rgb": diffuse_rgb + specular_rgb}
        return ret
