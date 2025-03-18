import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .models.embedder import get_embedder

# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=0,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            # layer paramaters initialization
            if geometric_init:
                if l == self.num_layers - 2: # Final layer
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else: # others
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        """
            return (sdf value, feature)
        """
        return self.forward(x)

    def gradient(self, x):
        """
            return normal vector
        """
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] # returns a list for all inputs
        return gradients

    def get_all(self, x, is_training=True):
        with torch.enable_grad():
            x.requires_grad_(True)
            tmp = self.forward(x)
            y, feature = tmp[..., :1], tmp[..., 1:]

            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True,
            )[0]
        if not is_training:
            return y.detach(), feature.detach(), gradients.detach()
        return y, feature, gradients


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        mode,
        d_in,
        d_feature, # semantic feature dim
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires=0,
        multires_view=0,
        squeeze_out=True,
        squeeze_out_scale=1.0,
        output_bias=0.0,
        output_scale=1.0,
        skip_in=(),
    ):
        """
        Settings:
            mode: "idr", "no_view_dir", "no_normal". `idr` input: ([points, view_dirs, normals, geo_f]).
            output_bias/scale: x = self.output_scale * (x + self.output_bias).
            squeeze_out:       x = self.squeeze_out_scale * torch.sigmoid(x).
            dout: 3.
        """
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        # dims: [tot_input_dims, hidden_layer_dims, ..., d_out]
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        # embed functions
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3 # replace pts with embed(pts)

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3 # replace view_dirs with embed(view_dirs)

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        self.output_bias = output_bias
        self.output_scale = output_scale
        self.squeeze_out_scale = squeeze_out_scale
        # self.last_active_fun = nn.Tanh()

    def forward(self, points, normals, view_dirs, geo_f, semantic_f=None):
        # input points/views -> embeded points/views
        if self.embed_fn is not None and points is not None:
            points = self.embed_fn(points)

        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == "idr":
            rendering_input = torch.cat([points, view_dirs, normals, geo_f], dim=-1)
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([points, normals, geo_f, semantic_f], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.output_scale * (x + self.output_bias)
        if self.squeeze_out:
            x = self.squeeze_out_scale * torch.sigmoid(x)

        return x
    
    @torch.enable_grad()
    def get_all(self, points, normals, view_dirs, geo_f, semantic_f=None, is_training=True):
        x = semantic_f
        x.requires_grad_(True)
        y = self.forward(points, normals, view_dirs, geo_f, x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        f_grads = torch.autograd.grad(
            outputs=y,
            inputs=semantic_f,
            grad_outputs=d_output,
            create_graph=is_training,
            retain_graph=is_training,
            only_inputs=True,
        )[0]
        if not is_training:
            return y.detach(), f_grads.detach()
        return y, f_grads


class PointLightNetwork(nn.Module):
    def __init__(self, enable_offset=False):
        super().__init__()
        self.register_parameter("light", nn.Parameter(torch.tensor(5.0)))
        if enable_offset:
            self.light_offset = nn.Embedding(1, 3)
            self.light_offset.weight.data = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]))

    def forward(self):
        return self.light
    
    @property
    def offset(self):
        return self.light_offset(torch.LongTensor([0]).cuda()) # [[x, y, z]]
    
    def set_light(self, light):
        self.light.data.fill_(light)

    def get_light(self):
        return self.light.data.detach()

    def get_offset(self):
        return self.light_offset.weight.data.detach()


class BlendNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires_view=8,
        squeeze_out=True,
        squeeze_out_scale=1.0,
        output_bias=0.0,
        output_scale=1.0,
        skip_in=(),
    ):
        super().__init__()

        self.squeeze_out = squeeze_out
        # [cos + roughness]
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, input_dims=1)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 1

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.output_bias = output_bias
        self.output_scale = output_scale
        self.squeeze_out_scale = squeeze_out_scale

    def forward(self, distance, roughness, dot_n_wo, features=None):
        if self.embedview_fn is not None:
            dot_n_wo = self.embedview_fn(dot_n_wo)
        rendering_input = None
        rendering_input = torch.cat([distance, roughness, dot_n_wo], dim=-1)
        if features is not None:
            rendering_input = torch.cat([rendering_input, features], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, rendering_input], dim=-1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.output_scale * (x + self.output_bias)
        if self.squeeze_out:
            x = self.squeeze_out_scale * torch.sigmoid(x)
        else:
            x = self.relu(x) * 2. * np.pi
        return x


class SingleVarianceNetwork(nn.Module):
    """
        optimize std deviation (s) of logistic density distribution, phi_s(x), `std = 1/s`
    """
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).cuda() * torch.exp(self.variance * 10.0)


class DINONetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(),
        multires=0,
        output_bias=0.0,
        output_scale=1.0,
        weight_norm=True,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] += input_ch - 3
        
        # self.embedview_fn = None
        # if multires_view > 0:
        #     embedview_fn, input_ch = get_embedder(multires_view)
        #     self.embedview_fn = embedview_fn
        #     dims[0] += input_ch - 3
        
        self.num_layers = len(dims)
        self.skip_in = skip_in
        
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                dims[l] += dims[0]
        
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(l), lin)
        
        self.relu = nn.ReLU()

        self.output_bias = output_bias
        self.output_scale = output_scale
        

    def forward(self, points, normals=None, features=None, dino_feature_split=None, is_training=False):
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        # if self.embedview_fn is not None and self.mode != "no_view_dir":
        #     view_dirs = self.embedview_fn(view_dirs)
        
        if normals is not None:
            if dino_feature_split is not None:
                input_x = torch.cat([points, normals, features, dino_feature_split], dim=-1)
            else:
                input_x = torch.cat([points, normals, features], dim=-1)
        else:
            input_x = torch.cat([points], dim=-1)
        x = input_x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input_x], dim=-1) / np.sqrt(2)
            
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.output_scale * (x + self.output_bias)
        # x = nn.functional.normalize(x, dim=-1)
        
        if not is_training:
            return x.detach()
        return x
    
    def passback(self, points, feature, is_training=True):
        with torch.enable_grad():
            x = points
            x.requires_grad_(True)
            
            y = feature
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=is_training,
                retain_graph=is_training,
                only_inputs=True,
            )[0]
        if not is_training:
            return y.detach(), gradients.detach()
        return y, gradients


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        d_in=3,
        d_in_view=3,
        multires=0,
        multires_view=0,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # -------- Main MLP --------
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)]
        )
        # -------- View MLP --------
        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False
