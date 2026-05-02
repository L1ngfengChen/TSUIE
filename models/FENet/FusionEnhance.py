import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
    

class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
     

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=96, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_chans=96, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.upsample = nn.Upsample(size=img_size, mode='bilinear', align_corners=False)

    def forward(self, x, x_size):
        B, HW, C = x.shape
        H, W = x_size
        
        assert HW == self.patches_resolution[0] * self.patches_resolution[1], \
            "Number of patches does not match the image size."

        x = x.transpose(1, 2).contiguous().view(B, C, self.patches_resolution[0], self.patches_resolution[1])

        x = self.upsample(x)

        return x
class HiLo(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)
        return x
    
    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)
        return x


class Cross_HiLo(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim
        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim
        self.ws = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.l_heads > 0:
            if self.ws != 1:
                self.x_sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.y_l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.x_l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.x_h_kv = nn.Linear(self.dim, self.h_dim * 2, bias=qkv_bias)
            self.y_h_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def c_hifi(self, x, y):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        y = y.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        x_kv = self.x_h_kv(x).reshape(B, total_groups, -1, 2, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        y_q = self.y_h_q(y).reshape(B, total_groups, -1, 1, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        x_k, x_v = x_kv[0], x_kv[1]
        y_q = y_q[0]

        attn = (y_q @ x_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ x_v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)

        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        x = self.h_proj(x)
        return x
    
    def c_lofi(self, x, y):
        B, H, W, C = x.shape
        y_q = self.y_l_q(y).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.x_sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_kv = self.x_l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            x_kv = self.x_l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        x_k, x_v = x_kv[0], x_kv[1]

        attn = (y_q @ x_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ x_v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x
    
    def forward(self, x, y):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.reshape(B, H, W, C)
        y = y.reshape(B, H, W, C)

        if self.h_heads == 0:
            x = self.c_lofi(x, y)
            return x.reshape(B, N, C)

        if self.l_heads == 0:
            x = self.c_hifi(x, y)
            return x.reshape(B, N, C)

        hifi_out = self.c_hifi(x, y)
        lofi_out = self.c_lofi(x, y)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_ws=2, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.att = HiLo(dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop=drop, window_size=local_ws, alpha=alpha)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.ffn = DWMlp(in_features=dim, hidden_features= mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.att(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    

class Cross_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_ws=2, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.att = Cross_HiLo(dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale, attn_drop = attn_drop, proj_drop=drop, window_size=local_ws, alpha=alpha)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.ffn = DWMlp(in_features=dim, hidden_features= mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        x = x + self.drop_path(self.att(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
    

class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, local_ws=2, alpha=0.5):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.blockx = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
        self.blocky = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
        self.Cx = Cross_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
        self.Cy = Cross_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
        
    def forward(self, x, y):
        x = self.blockx(x)
        y = self.blocky(y)
        x1 = self.Cx(x, y)
        y1 = self.Cy(y, x)
        return x1, y1
    

class DeepExLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0.,drop_path=0., norm_layer=nn.LayerNorm, local_ws=2, alpha=0.5):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        block = Block
        self.blocks = nn.ModuleList([
            block(dim=dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop,
                  attn_drop=attn_drop,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  norm_layer=norm_layer,
                  local_ws=local_ws,
                  alpha=alpha)
        for i in range(depth)])

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        return x


class HLFusionLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0.,drop_path=0., norm_layer=nn.LayerNorm, local_ws=2, alpha=0.5):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        block = CBlock
        self.blocks = nn.ModuleList([
            block(dim=dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop,
                  attn_drop=attn_drop,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  norm_layer=norm_layer,
                  local_ws=local_ws,
                  alpha=alpha)
        for i in range(depth)])

    def forward(self, x, y):
        for i, blk in enumerate(self.blocks):
            x, y = blk(x, y)
        return x, y


class FENet(nn.Module):
    def __init__(self,
                 image_size=256,
                 patch_size = 4,
                 in_channal = 3,
                 embed_dim = 96,
                 depths = [2, 4, 4, 2],
                 num_heads = [3, 6, 12, 24],
                 window_size = 8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 local_ws = [1, 2, 2, 1],
                 alpha = 0.5
                ):
        super().__init__()
        self.local_ws = local_ws
        self.alpha = alpha
        self.num_heads = num_heads
        self.pretrain_img_size = image_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, norm_layer=norm_layer, in_chans=embed_dim, embed_dim=embed_dim)
        self.patch_unembed = PatchUnEmbed(img_size=image_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer)
        self.softmax = nn.Softmax(dim=0)
        Ex_depths = depths[:1]
        Fusion_depths = depths[1:3]
        Re_depths = depths[3:]
        Ex_num_heads=num_heads[:1]
        Fusion_num_heads=num_heads[1:3]
        Re_num_heads=num_heads[3:]
        self.Ex_num_layers = len(Ex_depths) 
        self.Fusion_num_layers = len(Fusion_depths) 
        self.Re_num_layers = len(Re_depths)

        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.conv_first1_A = nn.Conv2d(in_channels=in_channal, out_channels=embed_dim//2, kernel_size=3, stride=1, padding=1)
        self.conv_first1_B = nn.Conv2d(in_channels=in_channal, out_channels=embed_dim//2, kernel_size=3, stride=1, padding=1)
        self.conv_first2_A = nn.Conv2d(embed_dim//2, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim//2, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = DeepExLayer(dim=embed_dim,
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         local_ws=local_ws[0],
                         alpha=alpha
                         )
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.embed_dim)

        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = DeepExLayer(dim=embed_dim,
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         local_ws=local_ws[0],
                         alpha=alpha
                         )
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.embed_dim)

        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = HLFusionLayer(dim=embed_dim,
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         local_ws=local_ws[1+i_layer],
                         alpha=alpha
                         )
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.embed_dim)
        self.norm_Fusion_B = norm_layer(self.embed_dim)
        self.Fusion_conv = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)

        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = DeepExLayer(dim=embed_dim,
                         depth=Re_depths[i_layer],
                         num_heads=Re_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, 
                         qk_scale=qk_scale,
                         drop=drop_rate, 
                         attn_drop=attn_drop_rate,
                         drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         local_ws=local_ws[3],
                         alpha=alpha
                         )
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.embed_dim)
        self.Last_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim//2, 3, 1, 1),
                                       self.lrelu,
                                       nn.Conv2d(embed_dim //2, embed_dim//4, 3, 1, 1),
                                       self.lrelu,
                                       nn.Conv2d(embed_dim // 4, in_channal, 3, 1, 1))
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
      
    def forward_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))           
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers_Ex_A:
            x = layer(x)

        x = self.norm_Ex_A(x)
        x = self.patch_unembed(x, x_size)
        return x
        
    def forward_Ex_B(self, x):
        x = self.lrelu(self.conv_first1_B(x))
        x = self.lrelu(self.conv_first2_B(x))           
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers_Ex_B:
            x = layer(x)

        x = self.norm_Ex_B(x)
        x = self.patch_unembed(x, x_size)
        return x
        
    def forward_Fusion(self, x, y):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        
        for layer in self.layers_Fusion:
            x, y = layer(x, y)
            
        x = self.norm_Fusion_A(x)
        x = self.patch_unembed(x, x_size)

        y = self.norm_Fusion_B(y)
        y = self.patch_unembed(y, x_size)
        x = torch.cat([x, y], 1)
        x = self.lrelu(self.Fusion_conv(x))
        return x

    def forward_Re(self, x):        
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x)

        x = self.norm_Re(x)
        x = self.patch_unembed(x, x_size)
        x = self.Last_conv(x)
        return x
    
    def forward(self, x, y):
        x = self.forward_Ex_A(x)
        y = self.forward_Ex_B(y)
        x = self.forward_Fusion(x, y)
        x = self.forward_Re(x)
        return x


if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.randn(1, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    print(x.shape)
    model = FENet(depths=[2, 2, 4, 2],num_heads=[2,4,8,4]).to(device)
    x = model(x, y)
    print(x.shape)