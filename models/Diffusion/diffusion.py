import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from inspect import isfunction
import time

def beta_schedule(start, end, n_timestep):
    betas = torch.linspace(start, end, n_timestep).double()
    return betas


def exists(x):
    return x is not None


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, opt):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.device = torch.device(opt.device)
        self.T = opt.T
        self.loss = nn.L1Loss().to(self.device)
        self.linear_start = opt.linear_start
        self.linear_end = opt.linear_end
        self.eta = 0


    def set_new_noise(self, linear_start, linear_end, n_timestep):
        betas = beta_schedule(linear_start, linear_end, n_timestep)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:n_timestep]

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', betas.to(self.device))
        self.register_buffer('alphas_bar', alphas_bar.to(self.device))
        self.register_buffer('alphas_bar_prev', alphas_bar_prev.to(self.device))
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).to(self.device))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1.0 - alphas_bar).to(self.device))
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1./alphas_bar).to(self.device))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1./alphas_bar-1).to(self.device))
        posterior_variance = betas * (1. - alphas_bar_prev)/(1. - alphas_bar)
        self.register_buffer('posterior_variance', posterior_variance.to(self.device))
        self.register_buffer('posterior_log_variance_clipped',torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20))).to(self.device))
        self.register_buffer('posterior_mean_coef1', (betas*torch.sqrt(alphas_bar_prev)/(1.-alphas_bar)).to(self.device))
        self.register_buffer('posterior_mean_coef2', ((1.-alphas_bar_prev)*torch.sqrt(1.-betas)/(1.-alphas_bar)).to(self.device))

    def predict_xt_prec_mean_from_noise(self,x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_bar,t, x_t.shape)*x_t - extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape)*noise)
    
    def q_posterior(self, x_start, x_t, t):
        mean = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var
    
    def p_mean_variance(self, x_t, y, t):
        x_recon = self.predict_xt_prec_mean_from_noise(x_t, t, noise = self.denoise_fn(torch.cat([x_t, y], dim = 1), t))
        x_recon=torch.clamp(x_recon, -1, 1)
        mean, var, log_var = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return mean, var, log_var
    
    def p_sample_ddim(self, x, label, t, t_next):
        noise = self.denoise_fn(torch.cat([x, label], dim = 1), t)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape).to(self.device)

        x0_t = (x - noise *torch.sqrt((1-at))) / torch.sqrt(at).to(self.device)
        if t_next == None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape).to(self.device)

        if self.eta == 0:
            xt_next = torch.sqrt(at_next) * x0_t + torch.sqrt(1 - at_next) * noise
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = self.eta * torch.sqrt((1 - at / (at_next)) * (1 - at_next) / (1 - at))
            c2 = torch.sqrt((1 - at_next) - c1**2)
            xt_next = torch.sqrt(at_next) * x0_t + c2 * noise + c1 * torch.rand_like(x0_t)

        return xt_next
        
    
    @torch.no_grad()
    def p_sample(self, x_t, y, t):
        self.set_new_noise(self.linear_start, self.linear_end, self.T)
        mean, _, log_var = self.p_mean_variance(x_t=x_t, y=y, t=t)
        b, *_, device = *x_t.shape, x_t.device
        noise = torch.randn_like(x_t).to(self.device)
        nonzero_mask = (1 - (t == 0).float()).reshape(b,*((1,) * (len(x_t.shape) - 1)))
        return mean + nonzero_mask*(0.5*log_var).exp()*noise
    
    @torch.no_grad()
    def ddim_sample(self, x, time_steps = None):
        self.set_new_noise(self.linear_start, self.linear_end, self.T)
        if time_steps==None:
            time_steps = np.array([1898, 1370, 340])
        
        label = x
        b = x.shape[0]
        img = torch.randn_like(label)

        for j, i in enumerate(time_steps):
            t = torch.full((b,), i, dtype=torch.long).to(self.device)
            if j == len(time_steps) - 1:
                t_next = None
            else:
                t_next = torch.full((b,), time_steps[j + 1], dtype=torch.long).to(self.device)

            img = self.p_sample_ddim(img, label, t, t_next)

        return torch.clamp(img, -1, 1)
    
    def forward(self, x, y , label=None, Training = True):
        self.set_new_noise(self.linear_start, self.linear_end, self.T)
        if Training:
            t = torch.randint(0, self.T, (x.shape[0],), device=self.device).long()
            noise = torch.randn_like(x)
            x_t = (extract(self.sqrt_alphas_bar, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_bar, t, x.shape) * noise)
            x_recon = self.denoise_fn(torch.cat([x_t, y], dim = 1), t)
            loss = self.loss(x_recon,noise)
            return loss
        else:
            x_t = x
            count = 0
            for time_step in reversed(range(self.T)):
                t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
                x_t = self.p_sample(x_t, y, t)
                count += 1


            x_0 = x_t
            return torch.clamp(x_0, -1, 1)