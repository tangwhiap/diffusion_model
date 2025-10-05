#!/usr/bin/env python
import os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

total_timesteps = 300
norm_groups = 8

img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]  # [64,128,256,512]
has_attention = [False, False, True, True]
num_res_blocks = 2

def load_model(model: nn.Module, path: str, map_location=None):
    blob = torch.load(path, map_location=map_location)
    model.load_state_dict(blob["state_dict"])
    return model

def swish(x): return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("emb", torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb))

    def forward(self, t):
        t = t.float()
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class TimeMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
    def forward(self, temb):
        return self.lin2(swish(self.lin1(temb)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x, temb):
        h = self.conv1(swish(self.norm1(x)))
        h = h + self.emb_proj(swish(temb))[:, :, None, None]
        h = self.conv2(swish(self.norm2(h)))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        x_n = self.norm(x)
        q = self.q(x_n).reshape(b, c, -1).transpose(1, 2)  # [B, HW, C]
        k = self.k(x_n).reshape(b, c, -1)                  # [B, C, HW]
        v = self.v(x_n).reshape(b, c, -1).transpose(1, 2)  # [B, HW, C]
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)

class DownSample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, ch): super().__init__(); self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x): return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))



class UNet(nn.Module):
    def __init__(self, img_size, img_channels, widths, has_attention,
                 num_res_blocks=2, norm_groups=8, first_conv_channels=64):
        super().__init__()
        self.widths = widths
        self.has_attention = has_attention
        self.num_res_blocks = num_res_blocks

        time_ch = first_conv_channels * 4
        self.time_embed = nn.Sequential(TimeEmbedding(time_ch), TimeMLP(time_ch, time_ch))
        self.input_conv = nn.Conv2d(img_channels, first_conv_channels, 3, padding=1)

        # -------- Down path --------
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_ch = first_conv_channels
        L = len(widths)
        for i, w in enumerate(widths):
            resblocks = nn.ModuleList()
            attnblocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                resblocks.append(ResidualBlock(in_ch, w, time_ch, groups=norm_groups))
                if has_attention[i]:
                    attnblocks.append(AttentionBlock(w, groups=norm_groups))
                in_ch = w
            self.down_res.append(resblocks)
            self.down_attn.append(attnblocks)
            self.downsample.append(DownSample(in_ch) if i != L - 1 else nn.Identity())

        # -------- Middle --------
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_ch, groups=norm_groups)
        self.mid_attn   = AttentionBlock(in_ch, groups=norm_groups)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_ch, groups=norm_groups)

        # -------- Up path --------
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # 我们在构建阶段就精确指定每个 up residual 的输入通道：in_ch + skip_ch
        curr_ch = in_ch  # 进入最底层 up 前的通道数（= widths[-1]）
        for i in reversed(range(L)):
            w = widths[i]
            prev_w = widths[i - 1] if i - 1 >= 0 else w  # 顶层没有更上一层，用 w 占位
            resblocks = nn.ModuleList()
            attnblocks = nn.ModuleList()

            # 对应 pop 顺序：先用当前 level 的 num_res_blocks 个 skip（通道数 = w），
            # 最后一次用来自上一层 DownSample 的 skip（通道数 = prev_w）
            skip_ch_seq = [w] * num_res_blocks + ([prev_w] if i > 0 else [w])

            for skip_ch in skip_ch_seq:
                resblocks.append(ResidualBlock(curr_ch + skip_ch, w, time_ch, groups=norm_groups))
                if has_attention[i]:
                    attnblocks.append(AttentionBlock(w, groups=norm_groups))
                curr_ch = w  # 经过这个 residual 后，通道数变为 w

            self.up_res.append(resblocks)
            self.up_attn.append(attnblocks)
            self.upsample.append(UpSample(curr_ch) if i != 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(norm_groups, curr_ch)
        self.out_conv = nn.Conv2d(curr_ch, img_channels, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_embed(t)
        skips = [self.input_conv(x)]
        hs = skips[0]

        # Down
        for i in range(len(self.widths)):
            for k in range(self.num_res_blocks):
                hs = self.down_res[i][k](hs, temb)
                if self.has_attention[i]:
                    hs = self.down_attn[i][k](hs)
                skips.append(hs)  # 每个 residual（带可选 attn）结束后 append 一次
            if i != len(self.widths) - 1:
                hs = self.downsample[i](hs)
                skips.append(hs)  # 下采样后再 append 一次（给 up 的 +1 次用）

        # Middle
        hs = self.mid_block1(hs, temb)
        hs = self.mid_attn(hs)
        hs = self.mid_block2(hs, temb)

        # Up（构建阶段已按 reversed 顺序组织）
        for j in range(len(self.widths)):
            i = len(self.widths) - 1 - j
            for k in range(self.num_res_blocks + 1):
                skip = skips.pop()
                # 空间尺寸对齐保护（理论上应一致）
                while skip.shape[2:] != hs.shape[2:]:
                    skip = skips.pop()
                hs = torch.cat([hs, skip], dim=1)
                hs = self.up_res[j][k](hs, temb)
                if self.has_attention[i]:
                    hs = self.up_attn[j][k](hs)
            if i != 0:
                hs = self.upsample[j](hs)

        return self.out_conv(swish(self.out_norm(hs)))

if torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon / Metal 加速
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # 回退 CPU
print(f"[Info] Using device: {device}")
network = UNet(img_size, img_channels, widths, has_attention,
               num_res_blocks=num_res_blocks, norm_groups=norm_groups,
               first_conv_channels=first_conv_channels).to(device)
ema_network = copy.deepcopy(network).to(device)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.9999).astype(np.float32)


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=300, clip_min=-1.0, clip_max=1.0):
        super().__init__()
        self.timesteps = int(timesteps)
        self.clip_min = clip_min
        self.clip_max = clip_max

        betas = cosine_beta_schedule(self.timesteps)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32, device=alphas.device),
             alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    @staticmethod
    def _extract(a, t, x_shape):
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise):
        return self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def predict_start_from_noise(self, x_t, t, noise):
        return self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        mean = self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
               self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        logv = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, logv

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
        model_mean, var, logv = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, var, logv

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        #model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x, t, clip_denoised)
        model_mean, model_variance, _ = self.p_mean_variance(pred_noise, x, t, clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

gdf_util = GaussianDiffusion(timesteps=total_timesteps).to(device)

@torch.no_grad()
def generate_images(ema_model, gdf_util, num_images=16, device="cuda"):
    ema_model.eval()
    samples = torch.randn(num_images, img_channels, img_size, img_size, device=device)
    #print(len(list(reversed(range(gdf_util.timesteps)))))
    print("Generating figures ...")
    for t in tqdm(list(reversed(range(gdf_util.timesteps)))):
        tt = torch.full((num_images,), t, device=device, dtype=torch.long)
        pred_noise = ema_model(samples, tt)
        samples = gdf_util.p_sample(pred_noise, samples, tt)
    return samples

def plot_images(samples, num_rows=2, num_cols=8, figsize=(12, 5)):
    samples = samples.clamp(-1, 1)
    samples = ((samples + 1) * 127.5).round().byte()
    samples = samples.permute(0, 2, 3, 1).cpu().numpy()
    _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i, img in enumerate(samples):
        ax[i // num_cols, i % num_cols].imshow(img)
        ax[i // num_cols, i % num_cols].axis("off")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def sample_and_plot(num_rows=2, num_cols=4, ckpt_path: str = ""):
    # 如果存在已训练好的 EMA 权重，优先加载
    if os.path.isfile(ckpt_path):
        load_model(ema_network, ckpt_path, map_location=device)
        ema_network.to(device)
        print(f"[Info] Loaded EMA weights from: {ckpt_path}")
    else:
        print("[Warn] No EMA checkpoint found; sampling with current (possibly untrained) weights.")

    samples = generate_images(ema_network, gdf_util,
                              num_images=num_rows * num_cols, device=device)
    plot_images(samples, num_rows=num_rows, num_cols=num_cols)


if __name__ == "__main__":

    epoch_check = 45
    CKPT_DIR = "models/epoch_%d" % (epoch_check)
    NET_CKPT = os.path.join(CKPT_DIR, "64x64_unet.pth")
    EMA_CKPT = os.path.join(CKPT_DIR, "64x64_unet_ema.pth")
    sample_and_plot(ckpt_path = EMA_CKPT, num_rows=2, num_cols=4)

