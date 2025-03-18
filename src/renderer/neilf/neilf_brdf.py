#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np

EPS = 1e-7

def neilf_rendering_equation(output_dirs,
                             distances,
                             normals,
                             base_color,
                             specular_albedo,
                             roughness,
                             incident_lights,
                             incident_dirs):
    """
        distance: [..., 1]
    """

    # extend all inputs into shape [N, 1, 1/3] for multiple incident lights
    # if incident_lights.dim() == 1:
    #     incident_lights = incident_lights.unsqueeze(dim=1).repeat(1, 3)                 # [1, 3]
    light_intensity = incident_lights / (distances * distances + 1e-10)                   # [N, 1]
    # output_dirs = output_dirs.unsqueeze(dim=1)                                          # [N, 3]
    normal_dirs = normals
    # normal_dirs = normals.unsqueeze(dim=1)                                              # [N, 3]
    # base_color = base_color.unsqueeze(dim=1)                                            # [N, 3]
    roughness = roughness.unsqueeze(dim=1)                                                # [N, 1]

    def _dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)                                          # [N, 1]

    def _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, roughness):

        return base_color / np.pi                                      # [N, 1, 1] # ? [N, 3]

    def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, specular_albedo):

        # used in SG, wrongly normalized
        def _d_sg(r, cos):
            r2 = (r * r).clamp(min=EPS)
            amp = 1 / (r2 * np.pi)
            sharp = 2 / r2
            return amp * torch.exp(sharp * (cos - 1))
        D = _d_sg(roughness, h_d_n)

        # Fresnel term F
        F_0 = 0.0                              # [N, 3]
        F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)                                      # [N, 1]

        # geometry term V, we use V = G / (4 * cos * cos) here
        def _v_schlick_ggx(r, cos):
            r2 = ((1 + r) ** 2) / 8
            return 0.5 / (cos * (1 - r2) + r2).clamp(min=EPS)
        V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)     

        return D * F * V * specular_albedo                                              # [N, 3]

    # half vector and all cosines
    # print(incident_dirs.shape, output_dirs.shape)
    half_dirs = incident_dirs + output_dirs                                             # [N, 3]
    half_dirs = Func.normalize(half_dirs, dim=-1)                                       # [N, 3]
    h_d_n = _dot(half_dirs, normal_dirs).clamp(min=0)                                   # [N, 1]
    h_d_o = _dot(half_dirs, output_dirs).clamp(min=0)                                   # [N, 1]
    n_d_i = _dot(normal_dirs, incident_dirs).clamp(min=0)                               # [N, 1]
    n_d_o = _dot(normal_dirs, output_dirs).clamp(min=0)                                 # [N, 1]

    f_d = _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, roughness)              # [N, 3]
    f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, specular_albedo)      # [N, 3]
    
    diffuse_rgb = (f_d * light_intensity * n_d_i)          # [N, 3]
    specular_rgb = (f_s * light_intensity * n_d_i)         # [N, 3]
    rgb = diffuse_rgb + specular_rgb                                                    # [N, 3]

    return {
        "rgb": rgb,
        "diffuse_rgb": diffuse_rgb,
        "specular_rgb": specular_rgb,
    }
