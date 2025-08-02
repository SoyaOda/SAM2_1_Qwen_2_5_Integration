"""
LISA-改 (LISA-kai) main model implementation.

This module integrates all components:
- Vision Encoder (Qwen2.5-VL)
- Adapter
- SAM 2.1 Mask Decoder
- Prompt Encoder
- Depth Head
- LLM Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import logging

from models.vision_encoder import VisionEncoder
from models.adapter import VisionToSAMAdapter, PromptEmbeddingAdapter
from models.mask_decoder import MaskDecoder
from models.prompt_encoder import PromptEncoder
from models.depth_head import DepthHead, compute_depth_loss
from models.llm_integration import LLMIntegration
from models.tokenizer_extension import ExtendedQwenTokenizer
from configs.model_config import LISAConfig

logger = logging.getLogger(__name__)


class LISAModel(nn.Module):
    """
    LISA-改: Language Instructed Segmentation Assistant (improved).
    
    Integrates Qwen2.5-VL vision encoder with SAM 2.1 mask decoder
    for language-guided segmentation with optional depth estimation.
    """
    
    def __init__(self, config: LISAConfig):
        super().__init__()
        self.config = config
        
        # Initialize tokenizer
        self.tokenizer = ExtendedQwenTokenizer(
            model_path=config.qwen_model_path,
            device=config.device,
        )
        
        # Vision Encoder
        self.vision_encoder = VisionEncoder(config.vision_encoder)
        
        # Adapter layers
        self.vision_adapter = VisionToSAMAdapter(config.adapter)
        self.prompt_adapter = PromptEmbeddingAdapter(
            input_dim=256,  # SAM default
            output_dim=config.adapter.output_dim,
        )
        
        # SAM components
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,  # SAM default before adaptation
            image_embedding_size=config.vision_encoder.grid_size,
            input_image_size=(config.vision_encoder.image_size,) * 2,
        )
        
        self.mask_decoder = MaskDecoder(config.sam_decoder)
        
        # Depth Head (disabled in v0, enabled in v1)
        self.depth_enabled = config.depth_head is not None and config.depth_head.enable
        if config.depth_head is not None:
            self.depth_head = DepthHead(config.depth_head)
        else:
            self.depth_head = None
        
        # LLM Integration
        self.llm_integration = LLMIntegration(
            config.llm_integration,
            self.tokenizer,
            config.qwen_model_path,
        )
        
        # Connect prompt encoder to LLM
        self.prompt_encoder.text_projection = nn.Linear(
            config.llm_integration.llm_hidden_dim,
            256,  # SAM prompt encoder dimension
        )
        
        # Loss weights
        self.loss_weights = {
            "text": config.training.text_loss_weight,
            "mask": config.training.mask_loss_weight,
            "iou": config.training.iou_loss_weight,
            "depth": config.training.depth_loss_weight,
        }
        
        # Register hooks for intermediate features
        self.vision_encoder.register_hooks()
        
        logger.info(f"Initialized LISA-改 model ({config.model_size})")
    
    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode images through vision encoder and adapter.
        
        Args:
            images: Input images (B, 3, H, W)
            
        Returns:
            Dict containing:
                - image_embeddings: SAM-ready features (B, 512, 64, 64)
                - vision_features: Original vision features (B, 1280, 64, 64)
                - intermediate_features: Hook features for high-res
        """
        # Encode through vision transformer
        vision_features = self.vision_encoder(images, return_features=True)
        
        # Get intermediate features for high-resolution
        intermediate_features = self.vision_encoder.get_intermediate_features()
        
        # Adapt to SAM dimension
        image_embeddings = self.vision_adapter(vision_features)
        
        return {
            "image_embeddings": image_embeddings,
            "vision_features": vision_features,
            "intermediate_features": intermediate_features,
        }
    
    def generate_masks(
        self,
        image_embeddings: torch.Tensor,
        prompt_embeddings: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate segmentation masks.
        
        Args:
            image_embeddings: Image features from adapter
            prompt_embeddings: Text-based prompt embeddings from LLM
            points: Point prompts (coords, labels)
            boxes: Box prompts
            masks: Mask prompts
            multimask_output: Whether to output multiple masks
            
        Returns:
            Dict containing:
                - masks: Predicted masks (B, K, H, W)
                - iou_pred: IoU predictions (B, K)
                - mask_tokens: Mask token embeddings (B, K, C)
        """
        # Encode prompts
        # For text embeddings from LLM, they are already projected to the right dimension
        if prompt_embeddings is not None:
            # Use prompt embeddings directly as sparse embeddings
            sparse_embeddings = prompt_embeddings
            dense_embeddings = None
        else:
            # Use regular prompt encoder for other prompt types
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=masks,
            )
        
        # Adapt prompt embeddings to mask decoder dimension
        sparse_embeddings = self.prompt_adapter(sparse_embeddings)
        
        # Generate masks
        masks, iou_pred, mask_tokens = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        
        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "mask_tokens": mask_tokens,
        }
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        gt_depths: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through LISA-改.
        
        Args:
            images: Input images (B, 3, H, W)
            input_ids: Text token IDs (B, seq_len)
            labels: Target token IDs for language modeling
            points: Point prompts
            boxes: Box prompts
            masks: Mask prompts
            gt_masks: Ground truth masks for training
            gt_depths: Ground truth depths for training
            multimask_output: Whether to output multiple masks
            
        Returns:
            Dict containing all outputs and losses
        """
        # Encode images
        image_features = self.encode_image(images)
        image_embeddings = image_features["image_embeddings"]
        
        # Extract vision summary for LLM
        # Use global average pooling of mask tokens as summary
        vision_summary = image_embeddings.mean(dim=[2, 3])  # (B, 512)
        
        # Estimate depth (only if enabled)
        depth_pred = None
        depth_summary = None
        if self.depth_enabled and self.depth_head is not None:
            try:
                depth_pred = self.depth_head(image_embeddings)
                if self.config.depth_head.use_depth_tokens:
                    depth_summary = depth_pred.mean(dim=[2, 3])  # (B, 512)
            except RuntimeError as e:
                # Depth estimation disabled or failed
                logger.warning(f"Depth estimation skipped: {e}")
                depth_pred = None
        
        # LLM forward pass
        llm_outputs = self.llm_integration(
            input_ids=input_ids,
            vision_summary=vision_summary,
            depth_summary=depth_summary,
            labels=labels,
            return_seg_prompts=True,
        )
        
        # Initialize outputs
        outputs = {
            "logits": llm_outputs["logits"],
        }
        
        # Only include depth if computed
        if depth_pred is not None:
            outputs["depth"] = depth_pred
        
        total_loss = 0.0
        
        # Language modeling loss
        if "loss" in llm_outputs and llm_outputs["loss"] is not None:
            outputs["loss_text"] = llm_outputs["loss"]
            total_loss += self.loss_weights["text"] * llm_outputs["loss"]
        
        # Generate masks if <SEG> tokens present
        if llm_outputs.get("seg_prompts") is not None:
            # Batch seg_prompts by batch index
            seg_prompts = llm_outputs["seg_prompts"]
            seg_positions = llm_outputs["seg_positions"]
            
            # For simplicity, take the first <SEG> token per batch
            # In practice, you'd handle multiple <SEG> tokens per batch
            B = image_embeddings.shape[0]
            batched_prompts = []
            
            for b in range(B):
                batch_mask = seg_positions[:, 0] == b
                if batch_mask.any():
                    # Take first <SEG> for this batch
                    batched_prompts.append(seg_prompts[batch_mask][0:1])
                else:
                    # No <SEG> token for this batch, use zero embedding
                    batched_prompts.append(torch.zeros(1, seg_prompts.shape[-1], device=seg_prompts.device, dtype=seg_prompts.dtype))
            
            batched_prompts = torch.stack(batched_prompts)  # (B, 1, C)
            
            mask_outputs = self.generate_masks(
                image_embeddings=image_embeddings,
                prompt_embeddings=batched_prompts,
                points=points,
                boxes=boxes,
                masks=masks,
                multimask_output=multimask_output,
            )
            
            outputs.update({
                "masks": mask_outputs["masks"],
                "iou_pred": mask_outputs["iou_pred"],
                "seg_positions": llm_outputs["seg_positions"],
            })
            
            # Compute mask losses if ground truth available
            if gt_masks is not None:
                mask_loss, iou_loss = self.compute_mask_losses(
                    mask_outputs["masks"],
                    mask_outputs["iou_pred"],
                    gt_masks,
                )
                outputs["loss_mask"] = mask_loss
                outputs["loss_iou"] = iou_loss
                
                total_loss += self.loss_weights["mask"] * mask_loss
                total_loss += self.loss_weights["iou"] * iou_loss
        
        # Depth loss (only if depth is enabled and predicted)
        if gt_depths is not None and depth_pred is not None:
            try:
                depth_loss = compute_depth_loss(depth_pred, gt_depths)
                outputs["loss_depth"] = depth_loss
                total_loss += self.loss_weights["depth"] * depth_loss
            except ValueError as e:
                logger.error(f"Depth loss computation failed: {e}")
                # Continue without depth loss
        
        # Total loss
        if total_loss > 0:
            outputs["loss"] = total_loss
        
        return outputs
    
    def compute_mask_losses(
        self,
        pred_masks: torch.Tensor,
        pred_ious: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute segmentation losses.
        
        Args:
            pred_masks: Predicted masks (B, K, H, W)
            pred_ious: Predicted IoUs (B, K)
            gt_masks: Ground truth masks (B, 1, H, W)
            
        Returns:
            mask_loss: Combined BCE and Dice loss
            iou_loss: IoU prediction MSE loss
        """
        B, K, H, W = pred_masks.shape
        
        # For single GT mask, select best prediction based on IoU
        if K > 1:
            # Compute actual IoUs with GT
            with torch.no_grad():
                actual_ious = []
                for k in range(K):
                    iou = self._compute_iou(pred_masks[:, k:k+1], gt_masks)
                    actual_ious.append(iou)
                actual_ious = torch.stack(actual_ious, dim=1)  # (B, K)
                
                # Select best mask
                best_idx = actual_ious.argmax(dim=1, keepdim=True)  # (B, 1)
                best_idx = best_idx.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
                best_idx = best_idx.expand(-1, -1, H, W)
                
                pred_masks = torch.gather(pred_masks, 1, best_idx)  # (B, 1, H, W)
                best_idx_iou = best_idx[:, :, 0, 0]  # (B, 1)
                pred_ious = torch.gather(pred_ious, 1, best_idx_iou)  # (B, 1)
        
        # Resize GT masks if needed
        if gt_masks.shape[-2:] != (H, W):
            gt_masks = F.interpolate(
                gt_masks.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )
        
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, gt_masks, reduction='mean'
        )
        
        # Dice loss
        pred_masks_sigmoid = torch.sigmoid(pred_masks)
        intersection = (pred_masks_sigmoid * gt_masks).sum(dim=[2, 3])
        union = pred_masks_sigmoid.sum(dim=[2, 3]) + gt_masks.sum(dim=[2, 3])
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        dice_loss = dice_loss.mean()
        
        # Combined mask loss
        mask_loss = bce_loss + dice_loss
        
        # IoU prediction loss
        with torch.no_grad():
            actual_iou = self._compute_iou(pred_masks_sigmoid, gt_masks)
        iou_loss = F.mse_loss(pred_ious.squeeze(), actual_iou.squeeze())
        
        return mask_loss, iou_loss
    
    def _compute_iou(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU between predictions and targets."""
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum(dim=[2, 3])
        union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) - intersection
        iou = intersection / (union + 1e-8)
        return iou
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        return_masks: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text response and segmentation masks.
        
        Args:
            images: Input images
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_masks: Whether to generate masks for <SEG> tokens
            
        Returns:
            Dict containing generated text and masks
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(images.device)
        
        # Encode images
        image_features = self.encode_image(images)
        vision_summary = image_features["image_embeddings"].mean(dim=[2, 3])
        
        # Generate text
        output_ids, seg_info = self.llm_integration.generate(
            input_ids=inputs["input_ids"],
            vision_summary=vision_summary,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Decode text
        generated_text = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        )
        
        results = {
            "text": generated_text,
            "seg_info": seg_info,
        }
        
        # Generate masks for each <SEG> token
        if return_masks and seg_info:
            masks = []
            for info in seg_info:
                # Get hidden state at <SEG> position
                # (Simplified - in practice, run another forward pass)
                mask_outputs = self.generate_masks(
                    image_embeddings=image_features["image_embeddings"][[info["batch_idx"]]],
                    multimask_output=False,
                )
                masks.append(mask_outputs["masks"][0])
            
            results["masks"] = masks
        
        return results