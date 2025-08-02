"""
Depth Estimation Head for LISA-改.

This module implements depth estimation capabilities,
starting with a dummy implementation for v0 that can be
easily replaced with real depth estimation in v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

from configs.model_config import DepthHeadConfig

logger = logging.getLogger(__name__)


class DepthHeadDummy(nn.Module):
    """
    Dummy depth head for v0 - returns zero tensors.
    
    This will be replaced with actual depth estimation in v1.
    """
    
    def __init__(self, config: DepthHeadConfig):
        super().__init__()
        self.config = config
        self.output_size = config.output_size
        
        logger.info("Initialized DepthHeadDummy (v0) - depth estimation disabled")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return zero depth map.
        
        Args:
            x: Image features (B, C, H, W)
            
        Returns:
            Zero depth map (B, 1, output_size, output_size)
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Return zeros with correct shape
        return torch.zeros(
            B, 1, self.output_size, self.output_size,
            device=device, dtype=dtype
        )


class DepthHead(nn.Module):
    """
    Depth Estimation Head for LISA-改.
    
    In v0: Acts as a placeholder returning zeros
    In v1: Will perform actual depth estimation
    """
    
    def __init__(self, config: DepthHeadConfig):
        super().__init__()
        self.config = config
        
        if not config.enable:
            # v0: Use dummy implementation
            self.depth_estimator = DepthHeadDummy(config)
        else:
            # v1: Use real implementation
            self.depth_estimator = DepthHeadReal(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depth head."""
        return self.depth_estimator(x)


class DepthHeadReal(nn.Module):
    """
    Real depth estimation head implementation for v1.
    
    Uses ConvTranspose layers to upsample features to depth map.
    """
    
    def __init__(self, config: DepthHeadConfig):
        super().__init__()
        self.config = config
        
        # Build upsampling network
        layers = []
        in_dim = config.input_dim
        
        # Progressive upsampling with hidden dimensions
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.ConvTranspose2d(
                    in_dim, hidden_dim,
                    kernel_size=2, stride=2
                ),
                nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
                nn.GELU() if config.use_gelu else nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim
        
        # Final layer to single channel depth
        layers.append(
            nn.Conv2d(in_dim, config.output_dim, kernel_size=1)
        )
        
        self.depth_layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized DepthHeadReal with output size {config.output_size}")
    
    def _init_weights(self):
        """Initialize depth head weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth map from features.
        
        Args:
            x: Image features (B, C, H, W)
            
        Returns:
            Depth map (B, 1, output_size, output_size)
        """
        # Apply depth estimation layers
        depth = self.depth_layers(x)
        
        # Ensure output size matches config
        if depth.shape[-2:] != (self.config.output_size, self.config.output_size):
            depth = F.interpolate(
                depth,
                size=(self.config.output_size, self.config.output_size),
                mode='bilinear',
                align_corners=False
            )
        
        return depth


class DepthTokenAdapter(nn.Module):
    """
    Adapter to convert depth features to LLM tokens (for v1.5).
    
    This module will be used to inject depth information
    into the LLM as special tokens.
    """
    
    def __init__(
        self,
        depth_dim: int = 512,
        llm_dim: int = 2048,
        num_tokens: int = 1
    ):
        super().__init__()
        self.num_tokens = num_tokens
        
        if num_tokens == 1:
            # Simple global pooling + projection
            self.adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(depth_dim, llm_dim),
                nn.LayerNorm(llm_dim),
            )
        else:
            # Multiple tokens via learned queries (Q-Former style)
            self.depth_queries = nn.Parameter(
                torch.randn(num_tokens, depth_dim)
            )
            self.cross_attention = nn.MultiheadAttention(
                depth_dim, num_heads=8, batch_first=True
            )
            self.projection = nn.Linear(depth_dim, llm_dim)
            self.norm = nn.LayerNorm(llm_dim)
    
    def forward(self, depth_features: torch.Tensor) -> torch.Tensor:
        """
        Convert depth features to LLM tokens.
        
        Args:
            depth_features: Depth features (B, C, H, W)
            
        Returns:
            Depth tokens for LLM (B, num_tokens, llm_dim)
        """
        B = depth_features.shape[0]
        
        if self.num_tokens == 1:
            # Simple pooling
            tokens = self.adapter(depth_features).unsqueeze(1)
        else:
            # Cross-attention with learned queries
            # Flatten spatial dimensions
            depth_flat = depth_features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            
            # Expand queries for batch
            queries = self.depth_queries.unsqueeze(0).expand(B, -1, -1)
            
            # Cross attention
            tokens, _ = self.cross_attention(queries, depth_flat, depth_flat)
            
            # Project to LLM dimension
            tokens = self.projection(tokens)
            tokens = self.norm(tokens)
        
        return tokens


def compute_depth_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    loss_type: str = "l1_silog",
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Compute depth estimation loss.
    
    Args:
        pred: Predicted depth map (B, 1, H, W)
        target: Ground truth depth map (B, 1, H, W)
        valid_mask: Valid pixel mask (B, 1, H, W)
        loss_type: Type of loss ("l1", "silog", "l1_silog")
        alpha: Weight for combining L1 and SiLog losses
        
    Returns:
        Depth loss value
    """
    # For v0, always return 0
    if target is None or torch.all(target == 0):
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    # Apply valid mask if provided
    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask
        n_valid = valid_mask.sum() + 1e-8
    else:
        n_valid = pred.numel()
    
    # L1 loss
    l1_loss = torch.abs(pred - target).sum() / n_valid
    
    if loss_type == "l1":
        return l1_loss
    
    # Scale-invariant logarithmic loss (SiLog)
    with torch.no_grad():
        # Avoid log(0)
        pred_log = torch.log(torch.clamp(pred, min=1e-8))
        target_log = torch.log(torch.clamp(target, min=1e-8))
    
    diff_log = pred_log - target_log
    
    if valid_mask is not None:
        diff_log = diff_log * valid_mask
    
    silog_loss = torch.sqrt(
        (diff_log ** 2).sum() / n_valid -
        0.5 * (diff_log.sum() / n_valid) ** 2
    )
    
    if loss_type == "silog":
        return silog_loss
    elif loss_type == "l1_silog":
        return l1_loss + alpha * silog_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")