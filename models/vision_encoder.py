"""
Vision Encoder module for LISA-改.

This module implements the Qwen2.5-VL inspired Vision Transformer (ViT)
with window attention and global attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math
from einops import rearrange
import logging

from configs.model_config import VisionEncoderConfig

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in Qwen models."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * dim * 4 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class WindowAttention(nn.Module):
    """Window-based Multi-Head Attention for efficient processing."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        
        # Reshape to image format
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Pad if necessary
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H_pad, W_pad = H + pad_h, W + pad_w
        else:
            H_pad, W_pad = H, W
        
        # Reshape to windows
        x = rearrange(
            x, 
            'b c (h_w h) (w_w w) -> (b h_w w_w) (h w) c',
            h_w=H_pad // self.window_size,
            w_w=W_pad // self.window_size,
            h=self.window_size,
            w=self.window_size
        )
        
        # Apply attention
        qkv = self.qkv(x).reshape(
            x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back
        x = rearrange(
            x,
            '(b h_w w_w) (h w) c -> b c (h_w h) (w_w w)',
            b=B,
            h_w=H_pad // self.window_size,
            w_w=W_pad // self.window_size,
            h=self.window_size,
            w=self.window_size
        )
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        # Back to sequence format
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        return x


class GlobalAttention(nn.Module):
    """Global Multi-Head Attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with either window or global attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        use_window_attn: bool = True,
        window_size: int = 8,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        
        if use_window_attn:
            self.attn = WindowAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
                dropout=dropout,
            )
        else:
            self.attn = GlobalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                dropout=dropout,
            )
        
        self.norm2 = RMSNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    
    def __init__(
        self,
        img_size: int = 896,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        x = self.proj(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class VisionEncoder(nn.Module):
    """
    Vision Encoder for LISA-改.
    
    Implements a Vision Transformer with mixed window/global attention
    following Qwen2.5-VL architecture principles.
    """
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_dim,
        )
        
        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_dim)
        )
        self.pos_drop = nn.Dropout(config.dropout_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            # Use global attention for specified layers
            use_global = i in config.global_attn_layers
            
            block = TransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                dropout=config.dropout_rate,
                use_window_attn=not use_global,
                window_size=config.local_attn_window_size,
            )
            self.blocks.append(block)
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
        # Hook handles for intermediate features
        self._hook_handles = []
        self._intermediate_features = {}
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        """Initialize weights for a module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        self._clear_hooks()
        
        for idx in self.config.hook_layers:
            handle = self.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hook_handles.append(handle)
    
    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            self._intermediate_features[layer_idx] = output
        return hook
    
    def _clear_hooks(self):
        """Clear all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._intermediate_features = {}
    
    def get_intermediate_features(self) -> Dict[int, torch.Tensor]:
        """Get captured intermediate features."""
        return self._intermediate_features.copy()
    
    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            x: Input images of shape (B, 3, H, W)
            return_features: Whether to return as 2D feature map
            
        Returns:
            If return_features is True: (B, C, H', W') feature map
            Otherwise: (B, N, C) sequence of patch features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        if return_features:
            # Reshape to 2D feature map
            H = W = int(math.sqrt(x.shape[1]))
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x
    
    def __del__(self):
        """Clean up hooks when object is deleted."""
        self._clear_hooks()