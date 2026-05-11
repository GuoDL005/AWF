# The complete code will be released after the journal accepts the article！！！！！

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.mlp(self.pool(x))
        return x * weights

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, 
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_map = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn_map

class AWF(nn.Module):
    def __init__(self, channels: int = 512, reduction: int = 16):
        super().__init__()
        
        # --- 路径 1: 局部细节路径 ---


        # --- 路径 2: 上下文路径 ---
        self.path2_conv_dilated = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False) # 使用 DW Conv
        self.path2_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.path2_spatial_attn = SpatialAttention(kernel_size=7)
        self.path2_relu = nn.ReLU(inplace=True)

        # --- 路径 3: 跨模态交互路径 ---
        self.path3_ca_vi = ChannelAttention(channels, reduction)
        self.path3_ca_ir = ChannelAttention(channels, reduction)
        self.path3_relu = nn.ReLU(inplace=True)

        # --- 自适应融合权重生成器 ---


        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        vi_feat, ir_feat = x
        B, C, H, W = vi_feat.shape

        # --- 计算路径 1: 局部细节 ---


        # --- 计算路径 2: 上下文 ---
        combined_p2 = vi_feat + ir_feat 
        context_p2 = self.path2_pw(self.path2_conv_dilated(combined_p2))
        context_p2 = self.path2_relu(context_p2)
        out_p2 = self.path2_spatial_attn(context_p2)

        # --- 计算路径 3: 跨模态交互 ---
        vi_attn_for_ir = self.path3_ca_vi.mlp(self.path3_ca_vi.pool(vi_feat))
        ir_attn_for_vi = self.path3_ca_ir.mlp(self.path3_ca_ir.pool(ir_feat))
        vi_interacted = vi_feat * ir_attn_for_vi 
        ir_interacted = ir_feat * vi_attn_for_ir 
        out_p3 = self.path3_relu(vi_interacted + ir_interacted)

        # --- 计算自适应融合权重 ---


        # --- 自适应加权融合 ---
        fused_intermediate = w1 * out_p1 + w2 * out_p2 + w3 * out_p3

        # --- 最终特征增强 ---
        fused_output = self.enhance(fused_intermediate)

        return fused_output
