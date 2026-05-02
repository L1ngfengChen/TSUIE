import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from pytorch_wavelets import DWTForward
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Bias_LayerNorm(nn.Module):
    def __init__(self,normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
    

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = Bias_LayerNorm(dim)

    def forward(self,x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)
    

class CustomSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, time_emb):
        for module in self.modules_list:
            if isinstance(module, Head_Trans):
                x = module(x, time_emb)
            else:
                x = module(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x
    

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class ChannelAttention(nn.Module):
    def __init__(self, n_heads,k_size):
        super().__init__()
        self.n_heads = n_heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size, padding=(k_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        heads = x.chunk(self.n_heads,dim=1)
        outputs = []
        for head in heads:
            y = self.avg_pool(head)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            out = head * y.expand_as(head)
            outputs.append(out)

        output = torch.cat(outputs, dim=1)
        return output
    

class Trans(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.ca = ChannelAttention(n_heads, 3)
        self.norm2 = LayerNorm(dim)
        self.glu = ConvolutionalGLU(dim, dim*2.66)

    def forward(self, x):
        x = x + self.ca(self.norm1(x))
        x = x + self.glu(self.norm2(x))
        return x
    

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=24, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
    

class HeadBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, norm_groups=24):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        )
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
    

class Head_Trans(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, norm_groups=24):
        super().__init__()
        self.head = HeadBlock(dim, dim_out, time_emb_dim, norm_groups=norm_groups)
        self.trans = nn.Sequential(*[Trans(dim,n_heads=3) for i in range(1)])

    def forward(self, x, time_emb):
        x = self.head(x, time_emb)
        x = self.trans(x)
        return x
    

class Unet(nn.Module):
    def __init__(self,
                 in_channel = 6,
                 out_channel = 3,
                 inner_channel = 48,
                 norm_groups=24):
        super().__init__()
        dim = inner_channel
        time_dim = inner_channel
        self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        self.Down1 = CustomSequential(
            nn.Conv2d(in_channel, dim, 3, 1, 1),
            Head_Trans(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups)
        )
        self.Down2 = CustomSequential(
            nn.Conv2d(dim, dim//2, 3, 1, 1),
            Down_wt(dim//2, dim*2),
            Head_Trans(dim=dim*2, dim_out=dim*2, time_emb_dim=time_dim, norm_groups=norm_groups)
        )
        self.Down3 = CustomSequential(
            nn.Conv2d(dim*2, (dim*2)//2, 3, 1, 1),
            Down_wt((dim*2)//2, dim*2**2),
            Head_Trans(dim=dim*2**2, dim_out=dim*2**2, time_emb_dim=time_dim, norm_groups=norm_groups)
        )
        self.Down4 = CustomSequential(
            nn.Conv2d(dim*2**2, (dim*2**2)//2, 3, 1, 1),
            Down_wt((dim*2**2)//2, dim*2**3),
            Head_Trans(dim=dim*2**3, dim_out=dim*2**3, time_emb_dim=time_dim, norm_groups=norm_groups)
        )

        self.conv3_up = nn.Sequential(nn.Conv2d(dim*2**3,(dim*2**3)*2, 3, 1, 1),
                                      nn.PixelShuffle(2))
        self.block3_up = Head_Trans(dim=dim*2**2, dim_out=dim*2**2, time_emb_dim=time_dim, norm_groups=norm_groups)
        self.conv2_up = nn.Sequential(nn.Conv2d(dim*2**2, (dim*2**2)*2, 3, 1, 1),
                                      nn.PixelShuffle(2))
        self.block2_up = Head_Trans(dim=dim*2, dim_out=dim*2, time_emb_dim=time_dim, norm_groups=norm_groups)
        self.conv1_up = nn.Sequential(nn.Conv2d(dim*2, (dim*2)*2, 3, 1, 1),
                                      nn.PixelShuffle(2))
        self.block1_up = Head_Trans(dim=dim*2, dim_out=dim*2, time_emb_dim=time_dim, norm_groups=norm_groups)

        self.cat3 = nn.Conv2d(dim*2**3,dim*2**2,kernel_size=1)
        self.cat2 = nn.Conv2d(dim*2**2, dim*2,kernel_size=1)
        
        self.refine = Head_Trans(dim = dim*2, dim_out=dim*2, time_emb_dim=time_dim, norm_groups=norm_groups)
        self.de_predict = nn.Conv2d(dim*2, out_channel,kernel_size=1)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x1 = self.Down1(x, t)
        x2 = self.Down2(x1, t)
        x3 = self.Down3(x2, t)
        x4 = self.Down4(x3, t)
        de3 = self.conv3_up(x4)
        de3 = self.cat3(torch.cat([de3,x3], dim=1))
        de3 = self.block3_up(de3, t)

        de2 = self.conv2_up(de3)
        de2 = self.cat2(torch.cat([de2,x2],dim=1))
        de2 = self.block2_up(de2, t)

        de1 = self.conv1_up(de2)
        de1 = self.block1_up(torch.cat([de1, x1], dim=1),t)

        x_refine = self.refine(de1, t)
        x_out = self.de_predict(x_refine)
        return x_out
    

if __name__ == '__main__':
    x = torch.randn(2, 6, 256, 256)
    model = Unet()
    time = torch.tensor([1,2])
    output = model(x,time)
    print(output.shape)