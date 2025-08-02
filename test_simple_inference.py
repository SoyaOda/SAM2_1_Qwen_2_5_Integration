"""
Simple test for LISA-改 inference without full model.

This script tests basic components integration.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.model_config import LISAConfig, VisionEncoderConfig, AdapterConfig, SAMDecoderConfig
from models.vision_encoder import VisionEncoder
from models.adapter import VisionToSAMAdapter
from models.mask_decoder import MaskDecoder
from models.prompt_encoder import PromptEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_simple_inference():
    """Test simple inference pipeline."""
    logger.info("Testing simple inference pipeline...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create configs
    vision_config = VisionEncoderConfig(
        image_size=224,  # Smaller for testing
        patch_size=14,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    adapter_config = AdapterConfig(
        input_dim=768,
        output_dim=256,
    )
    
    sam_config = SAMDecoderConfig(
        transformer_dim=256,
        transformer_heads=8,
        transformer_layers=2,
        num_mask_tokens=4,
        final_mask_size=224,
    )
    
    # Create models
    vision_encoder = VisionEncoder(vision_config).to(device)
    adapter = VisionToSAMAdapter(adapter_config).to(device)
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(16, 16),  # 224/14
        input_image_size=(224, 224),
    ).to(device)
    mask_decoder = MaskDecoder(sam_config).to(device)
    
    # Create dummy input
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Vision encoding
    logger.info("Encoding image...")
    with torch.no_grad():
        vision_features = vision_encoder(dummy_image, return_features=True)
        logger.info(f"Vision features shape: {vision_features.shape}")
        
        # Adapt features
        adapted_features = adapter(vision_features)
        logger.info(f"Adapted features shape: {adapted_features.shape}")
        
        # Create prompt embeddings (no prompt)
        sparse_prompts, dense_prompts = prompt_encoder()
        logger.info(f"Sparse prompts shape: {sparse_prompts.shape}")
        
        # Decode masks
        masks, iou_pred, mask_tokens = mask_decoder(
            image_embeddings=adapted_features,
            sparse_prompt_embeddings=sparse_prompts,
            multimask_output=False,
        )
        logger.info(f"Masks shape: {masks.shape}")
        logger.info(f"IoU predictions: {iou_pred}")
    
    logger.info("✓ Simple inference test passed!")
    return True


if __name__ == "__main__":
    test_simple_inference()