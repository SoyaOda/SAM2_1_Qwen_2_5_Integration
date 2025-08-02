"""
SAM 2.1 Mask Decoder implementation for LISA-改.

This module implements the mask decoder following SAM 2.1 architecture
with two-way transformer and multi-mask output support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import logging
from einops import rearrange

from configs.model_config import SAMDecoderConfig

logger = logging.getLogger(__name__)


class TwoWayTransformer(nn.Module):
    """
    Two-way transformer for processing image and prompt features.
    
    This transformer allows bidirectional attention between
    prompt tokens and image features.
    """
    
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 512,
        mlp_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.layers = nn.ModuleList([
            TwoWayTransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Final attention from prompt to image
        self.final_attn_token_to_image = CrossAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-way transformer.
        
        Args:
            image_embedding: Image features (B, H*W, C)
            image_pe: Positional encoding for image (B, H*W, C)
            point_embedding: Prompt embeddings (B, N, C)
            
        Returns:
            queries: Processed prompt embeddings (B, N, C)
            keys: Processed image features (B, H*W, C)
        """
        # Input shapes are already correct: (B, H*W, C) and (B, N, C)
        # No need to reshape - SAM2 keeps them as sequences
        queries = point_embedding
        keys = image_embedding
        
        # Apply transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        # Apply final attention layer
        q = queries + point_embedding
        k = keys + image_pe
        
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        
        return queries, keys


class TwoWayTransformerBlock(nn.Module):
    """Single block of the two-way transformer."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = SelfAttention(embedding_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image = CrossAttention(
            embedding_dim, num_heads, dropout
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = MLPBlock(embedding_dim, mlp_dim, dropout)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(
            embedding_dim, num_heads, dropout
        )
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention on queries
        attn_out = self.self_attn(q=queries, k=queries, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)
        
        # Cross attention from queries to keys
        attn_out = self.cross_attn_token_to_image(
            q=queries + query_pe,
            k=keys + key_pe,
            v=keys,
        )
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # MLP on queries
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        # Cross attention from keys to queries
        attn_out = self.cross_attn_image_to_token(
            q=keys + key_pe,
            k=queries + query_pe,
            v=queries,
        )
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys


class SelfAttention(nn.Module):
    """Self-attention module."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.internal_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Reshape for multi-head attention
        batch_size = q.shape[0]
        q = q.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.internal_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embedding_dim)
        out = self.out_proj(out)
        
        return out


class CrossAttention(nn.Module):
    """Cross-attention module."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.internal_dim = embedding_dim // num_heads
        
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if v is None:
            v = k
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Reshape for multi-head attention
        batch_size = q.shape[0]
        q = q.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.internal_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.internal_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embedding_dim)
        out = self.out_proj(out)
        
        return out


class MLPBlock(nn.Module):
    """MLP block for transformer."""
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class MaskDecoder(nn.Module):
    """
    SAM 2.1 Mask Decoder.
    
    Predicts masks given image and prompt embeddings.
    """
    
    def __init__(self, config: SAMDecoderConfig):
        super().__init__()
        self.config = config
        self.transformer_dim = config.transformer_dim
        self.num_mask_tokens = config.num_mask_tokens
        
        # Transformer
        self.transformer = TwoWayTransformer(
            depth=config.transformer_layers,
            embedding_dim=config.transformer_dim,
            mlp_dim=2048,
            num_heads=config.transformer_heads,
        )
        
        # IoU prediction head
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(config.transformer_dim, config.iou_prediction_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.iou_prediction_head_hidden_dim, config.iou_prediction_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.iou_prediction_head_hidden_dim, 1),
        )
        
        # Mask tokens
        self.mask_tokens = nn.Embedding(config.num_mask_tokens, config.transformer_dim)
        
        # Output upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                config.transformer_dim,
                config.transformer_dim // 4,
                kernel_size=2,
                stride=2,
            ),
            nn.LayerNorm((config.transformer_dim // 4,)),
            nn.GELU(),
            nn.ConvTranspose2d(
                config.transformer_dim // 4,
                config.transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
        )
        
        # Hypernetwork MLPs for mask prediction
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.transformer_dim, config.transformer_dim),
                nn.ReLU(),
                nn.Linear(config.transformer_dim, config.transformer_dim),
            )
            for _ in range(config.num_mask_tokens)
        ])
        
        # IoU token
        self.iou_token = nn.Embedding(1, config.transformer_dim)
        
        # Positional encoding
        self.pe_layer = PositionEmbeddingRandom(config.transformer_dim // 2)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through mask decoder.
        
        Args:
            image_embeddings: Image features (B, C, H, W)
            sparse_prompt_embeddings: Prompt embeddings (B, N, C)
            dense_prompt_embeddings: Dense features for high-res (optional)
            multimask_output: Whether to output multiple masks
            
        Returns:
            masks: Predicted masks (B, K, H_out, W_out)
            iou_pred: IoU predictions (B, K)
            mask_tokens_out: Output mask tokens (B, K, C)
        """
        B, C, H, W = image_embeddings.shape
        
        # Get mask and IoU tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        iou_token = self.iou_token.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Concatenate output tokens
        output_tokens = torch.cat([iou_token, mask_tokens], dim=1)
        
        # Concatenate with prompt embeddings
        if sparse_prompt_embeddings.shape[1] > 0:
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        else:
            tokens = output_tokens
        
        # Reshape and create positional encoding
        image_embeddings_reshaped = image_embeddings.flatten(2).permute(0, 2, 1)
        # pe_layer returns (C, H, W) where C = 2 * num_pos_feats
        image_pe = self.pe_layer((H, W)).to(image_embeddings.device)
        # Add batch dimension and reshape to match image embeddings
        image_pe = image_pe.unsqueeze(0).expand(B, -1, -1, -1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        
        # Pass through transformer
        hs, _ = self.transformer(
            image_embeddings_reshaped,
            image_pe,
            tokens,
        )
        
        # Split output tokens
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:self.num_mask_tokens+1, :]
        
        # Predict masks using hypernetworks
        upscaled_embedding = image_embeddings
        hyper_in_list = []
        
        for i, mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(mlp(mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)
        B, K, C = hyper_in.shape
        
        # Generate masks
        masks = (hyper_in @ upscaled_embedding.flatten(2)).view(B, K, H, W)
        
        # Upscale masks
        if self.config.use_high_res_features and dense_prompt_embeddings is not None:
            # Combine with high-res features if available
            masks = self.output_upscaling(masks.flatten(0, 1)).view(
                B, K, self.config.final_mask_size, self.config.final_mask_size
            )
            # Add dense prompt embeddings if provided
            if dense_prompt_embeddings is not None:
                masks = masks + dense_prompt_embeddings.unsqueeze(1)
        else:
            # Simple upscaling
            masks = F.interpolate(
                masks,
                size=(self.config.final_mask_size, self.config.final_mask_size),
                mode='bilinear',
                align_corners=False,
            )
        
        # Predict IoU
        iou_pred = self.iou_prediction_head(iou_token_out).squeeze(-1)
        iou_pred = iou_pred.unsqueeze(1).expand(-1, K)
        
        # Select masks based on multimask_output
        if not multimask_output:
            # Return best single mask
            best_idx = iou_pred.argmax(dim=1, keepdim=True)
            masks = torch.gather(masks, 1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, masks.shape[2], masks.shape[3]
            ))
            iou_pred = torch.gather(iou_pred, 1, best_idx)
            mask_tokens_out = torch.gather(
                mask_tokens_out, 1, best_idx.unsqueeze(-1).expand(-1, -1, C)
            )
        
        return masks, iou_pred, mask_tokens_out


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """
    
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for given size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        # Create grid of normalized coordinates
        y_embed = torch.arange(h, device=device).float() / (h - 1) * 2 - 1
        x_embed = torch.arange(w, device=device).float() / (w - 1) * 2 - 1
        
        # Create meshgrid
        y_embed = y_embed.unsqueeze(1).repeat(1, w)
        x_embed = x_embed.unsqueeze(0).repeat(h, 1)
        
        # Stack to create (h, w, 2) grid
        grid = torch.stack([x_embed, y_embed], dim=-1)
        grid = grid.reshape(h * w, 2)
        
        # Project to frequency space
        # Ensure same dtype as the gaussian matrix
        grid = grid.to(self.positional_encoding_gaussian_matrix.dtype)
        pos_embed = grid @ self.positional_encoding_gaussian_matrix
        pos_embed = 2 * torch.pi * pos_embed
        
        # Apply sin/cos and concatenate
        pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
        pos_embed = pos_embed.reshape(h, w, -1).permute(2, 0, 1)
        
        return pos_embed