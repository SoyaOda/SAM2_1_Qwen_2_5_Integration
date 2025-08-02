"""
Adapter module for bridging Vision Encoder and SAM Decoder.

This module implements the dimensional projection and normalization
to convert Qwen2.5-VL vision features to SAM-compatible features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

from configs.model_config import AdapterConfig

logger = logging.getLogger(__name__)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Implementation based on timm library.
    """
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        
        return x * random_tensor
    
    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'


class VisionToSAMAdapter(nn.Module):
    """
    Adapter to convert vision encoder features to SAM decoder compatible format.
    
    This adapter performs:
    1. Dimensional projection (1280 -> 512)
    2. Normalization
    3. DropPath regularization
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        
        # 1x1 convolution for dimension projection
        self.conv_proj = nn.Conv2d(
            config.input_dim,
            config.output_dim,
            kernel_size=1,
            bias=True
        )
        
        # Normalization layer
        # Using GroupNorm with 1 group to simulate LayerNorm for 2D features
        if config.use_layer_norm:
            self.norm = nn.GroupNorm(
                num_groups=1,
                num_channels=config.output_dim,
                eps=1e-6
            )
        else:
            self.norm = nn.Identity()
        
        # DropPath for regularization
        self.drop_path = DropPath(config.drop_path_rate)
        
        # Optional residual connection if dimensions match
        self.use_residual = (config.input_dim == config.output_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized VisionToSAMAdapter: {config.input_dim} -> {config.output_dim} dim"
        )
    
    def _init_weights(self):
        """Initialize adapter weights."""
        # Initialize conv weights with small variance
        nn.init.xavier_uniform_(self.conv_proj.weight)
        if self.conv_proj.bias is not None:
            nn.init.constant_(self.conv_proj.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Vision features of shape (B, C_in, H, W)
            
        Returns:
            Adapted features of shape (B, C_out, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.config.input_dim, \
            f"Input channels {C} doesn't match config {self.config.input_dim}"
        
        # Store input for potential residual connection
        identity = x
        
        # Dimension projection
        x = self.conv_proj(x)
        
        # Normalization
        x = self.norm(x)
        
        # DropPath regularization
        x = self.drop_path(x)
        
        # Add residual if dimensions match
        if self.use_residual:
            x = x + identity
        
        return x


class PromptEmbeddingAdapter(nn.Module):
    """
    Adapter for prompt embeddings to match mask decoder dimensions.
    
    SAM's original prompt encoder outputs 256-dim embeddings,
    but our mask decoder uses 512-dim, so we need projection.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dim is None:
            # Simple linear projection
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            # Two-layer MLP with activation
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        if isinstance(self.proj, nn.Linear):
            nn.init.xavier_uniform_(self.proj.weight)
            nn.init.constant_(self.proj.bias, 0)
        else:
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project prompt embeddings to higher dimension.
        
        Args:
            x: Prompt embeddings of shape (B, N, input_dim)
            
        Returns:
            Projected embeddings of shape (B, N, output_dim)
        """
        return self.proj(x)


class MultiScaleFeatureAdapter(nn.Module):
    """
    Adapter for multi-scale features from vision encoder hooks.
    
    This processes intermediate features from different layers
    for use in high-resolution mask prediction.
    """
    
    def __init__(
        self,
        input_dims: list,
        output_dim: int = 256,
        target_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.target_size = target_size
        
        # Create projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, output_dim, kernel_size=1)
            for in_dim in input_dims
        ])
        
        # Normalization for each scale
        self.norms = nn.ModuleList([
            nn.GroupNorm(num_groups=1, num_channels=output_dim)
            for _ in input_dims
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0)
    
    def forward(self, features: list) -> list:
        """
        Process multi-scale features.
        
        Args:
            features: List of feature maps at different scales
            
        Returns:
            List of processed feature maps with consistent dimensions
        """
        assert len(features) == len(self.input_dims), \
            f"Expected {len(self.input_dims)} features, got {len(features)}"
        
        processed_features = []
        
        for feat, proj, norm in zip(features, self.projections, self.norms):
            # Project to common dimension
            x = proj(feat)
            x = norm(x)
            
            # Resize to target size if specified
            if self.target_size is not None:
                x = F.interpolate(
                    x,
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            processed_features.append(x)
        
        return processed_features