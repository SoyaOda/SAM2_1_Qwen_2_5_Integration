"""
Prompt Encoder for LISA-改.

This module encodes various types of prompts (points, boxes, masks, text)
into embeddings that can be processed by the mask decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PromptEncoder(nn.Module):
    """
    Encodes prompts for SAM mask decoder.
    
    Supports:
    - Point prompts (positive/negative clicks)
    - Box prompts (bounding boxes)
    - Mask prompts (coarse masks)
    - Text prompts (via embedding projection)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (896, 896),
        mask_in_chans: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        
        # Positional encoding for points and boxes
        self.pe_layer = PositionalEncodingGaussian(embed_dim // 2)
        
        # Point encoding
        self.num_point_embeddings = 4  # pos/neg + box corners
        self.point_embeddings = nn.Embedding(self.num_point_embeddings, embed_dim)
        
        # Mask encoding
        self.mask_in_chans = mask_in_chans
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.LayerNorm((mask_in_chans // 4,)),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.LayerNorm((mask_in_chans,)),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        
        # No prompt embedding (for automatic segmentation)
        self.no_prompt_embed = nn.Embedding(1, embed_dim)
        
        # Text prompt projection (for future integration)
        self.text_projection = None  # Will be set externally if needed
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize encoder weights."""
        # Initialize embeddings
        nn.init.normal_(self.point_embeddings.weight, std=0.02)
        nn.init.normal_(self.no_prompt_embed.weight, std=0.02)
        
        # Initialize conv layers
        for m in self.mask_downscaling.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Get positional encoding for the entire image.
        
        Returns:
            Positional encoding (1, embed_dim, H, W)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def encode_points(
        self,
        points: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        pad: bool = True,
    ) -> torch.Tensor:
        """
        Encode point prompts.
        
        Args:
            points: Point coordinates (B, N, 2) in range [0, 1]
            labels: Point labels (B, N) - 1 for positive, 0 for negative
            pad: Whether to pad output to fixed size
            
        Returns:
            Point embeddings (B, N, embed_dim)
        """
        if points is None or points.shape[1] == 0:
            return self.no_prompt_embed.weight.unsqueeze(0).expand(
                points.shape[0] if points is not None else 1, -1, -1
            )
        
        # Scale points to image embedding size
        points = points * torch.tensor(
            self.image_embedding_size, device=points.device, dtype=points.dtype
        )
        
        # Get positional encoding for points
        point_pe = self.pe_layer.forward_with_coords(
            points, self.image_embedding_size
        )
        
        # Get point embeddings based on labels
        point_embeddings = self.point_embeddings(labels)
        
        # Combine with positional encoding
        point_embeddings = point_embeddings + point_pe
        
        return point_embeddings
    
    def encode_boxes(
        self,
        boxes: Optional[torch.Tensor],
        pad: bool = True,
    ) -> torch.Tensor:
        """
        Encode box prompts.
        
        Args:
            boxes: Box coordinates (B, N, 4) as [x0, y0, x1, y1] in range [0, 1]
            pad: Whether to pad output
            
        Returns:
            Box embeddings (B, N*2, embed_dim) - 2 points per box
        """
        if boxes is None or boxes.shape[1] == 0:
            return self.no_prompt_embed.weight.unsqueeze(0).expand(
                boxes.shape[0] if boxes is not None else 1, -1, -1
            )
        
        B, N, _ = boxes.shape
        
        # Convert boxes to corner points
        corners = torch.stack([
            boxes[:, :, :2],  # Top-left
            boxes[:, :, 2:],  # Bottom-right
        ], dim=2).reshape(B, N * 2, 2)
        
        # Use corner embeddings (indices 2 and 3)
        corner_labels = torch.tensor(
            [2, 3], device=boxes.device, dtype=torch.long
        ).repeat(B, N)
        
        return self.encode_points(corners, corner_labels, pad=pad)
    
    def encode_masks(
        self,
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode mask prompts.
        
        Args:
            masks: Binary masks (B, 1, H, W)
            
        Returns:
            Mask embeddings (B, embed_dim, H', W')
        """
        if masks is None:
            return None
        
        # Downscale masks
        mask_embeddings = self.mask_downscaling(masks)
        
        return mask_embeddings
    
    def encode_text(
        self,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text prompts (for LISA-style integration).
        
        Args:
            text_embeddings: Text embeddings from LLM (B, N, text_dim)
            
        Returns:
            Projected embeddings (B, N, embed_dim)
        """
        if self.text_projection is None:
            raise ValueError(
                "Text projection not initialized. "
                "Set prompt_encoder.text_projection before using text prompts."
            )
        
        return self.text_projection(text_embeddings)
    
    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all prompts.
        
        Args:
            points: Tuple of (coords, labels) or None
            boxes: Box coordinates or None
            masks: Binary masks or None
            text_embeddings: Text embeddings or None
            
        Returns:
            sparse_embeddings: (B, N, embed_dim) - point/box/text embeddings
            dense_embeddings: (B, embed_dim, H, W) - mask embeddings
        """
        sparse_embeddings = []
        
        # Encode points
        if points is not None:
            coords, labels = points
            point_embeddings = self.encode_points(coords, labels)
            sparse_embeddings.append(point_embeddings)
        
        # Encode boxes
        if boxes is not None:
            box_embeddings = self.encode_boxes(boxes)
            sparse_embeddings.append(box_embeddings)
        
        # Encode text
        if text_embeddings is not None:
            text_prompt_embeddings = self.encode_text(text_embeddings)
            sparse_embeddings.append(text_prompt_embeddings)
        
        # Concatenate sparse embeddings
        if sparse_embeddings:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            # No prompts - use no_prompt embedding
            sparse_embeddings = self.no_prompt_embed.weight.unsqueeze(0)
        
        # Encode masks
        dense_embeddings = self.encode_masks(masks) if masks is not None else None
        
        return sparse_embeddings, dense_embeddings


class PositionalEncodingGaussian(nn.Module):
    """
    Positional encoding using Gaussian random spatial frequencies.
    """
    
    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid.
        
        Args:
            size: (H, W) size of the grid
            
        Returns:
            Positional encoding (embed_dim, H, W)
        """
        H, W = size
        device = self.positional_encoding_gaussian_matrix.device
        
        grid = torch.stack([
            torch.arange(W, device=device, dtype=torch.float32) / W,
            torch.arange(H, device=device, dtype=torch.float32) / H,
        ], dim=-1)
        grid = grid.reshape(H * W, 2)
        
        # Project to frequency space
        pos_embed = 2 * np.pi * grid @ self.positional_encoding_gaussian_matrix
        
        # Apply sin and cos
        pos_embed = torch.cat([torch.sin(pos_embed), torch.cos(pos_embed)], dim=-1)
        
        # Reshape to image
        return pos_embed.reshape(H, W, -1).permute(2, 0, 1)
    
    def forward_with_coords(
        self,
        coords: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Generate positional encoding for specific coordinates.
        
        Args:
            coords: Coordinates (B, N, 2)
            size: Image size for normalization
            
        Returns:
            Positional encoding (B, N, embed_dim)
        """
        B, N, _ = coords.shape
        H, W = size
        
        # Normalize coordinates to [0, 1]
        coords = coords.clone()
        coords[:, :, 0] = coords[:, :, 0] / W
        coords[:, :, 1] = coords[:, :, 1] / H
        
        # Project to frequency space
        coords = coords.reshape(B * N, 2)
        pos_embed = 2 * np.pi * coords @ self.positional_encoding_gaussian_matrix
        
        # Apply sin and cos
        pos_embed = torch.cat([torch.sin(pos_embed), torch.cos(pos_embed)], dim=-1)
        
        return pos_embed.reshape(B, N, -1)