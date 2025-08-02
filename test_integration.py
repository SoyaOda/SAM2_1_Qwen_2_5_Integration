"""
Integration test for LISA-改 model.

Tests the full model integration with mock components.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.model_config import LISAConfig
from models.lisa_model import LISAModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_integration():
    """Test full model integration."""
    logger.info("Testing LISA-改 full integration...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create small config for testing
    config = LISAConfig(model_size="3B")
    config.vision_encoder.image_size = 224  # Smaller for testing
    config.vision_encoder.patch_size = 14
    config.vision_encoder.hidden_dim = 768
    config.vision_encoder.num_heads = 12  # 768/12 = 64
    config.vision_encoder.num_layers = 12  # Fewer layers for testing
    config.vision_encoder.global_attn_layers = [2, 5, 8, 11]  # Adjust for 12 layers
    config.vision_encoder.hook_layers = [2, 5, 8, 11]  # Adjust for 12 layers
    config.adapter.input_dim = 768
    config.adapter.output_dim = 256
    config.sam_decoder.transformer_dim = 256
    config.sam_decoder.final_mask_size = 224
    # Fix LLM integration dimensions
    config.llm_integration.mask_token_proj_dim = 256  # Match SAM decoder dim
    
    # Disable depth for v0
    if config.depth_head:
        config.depth_head.enable = False
    
    # Create model
    logger.info("Creating LISA-改 model...")
    try:
        model = LISAModel(config)
        # Convert to half precision to match LLM
        model = model.half()
        model = model.to(device)
        model.eval()
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False
    
    # Test forward pass
    logger.info("Testing forward pass...")
    try:
        with torch.no_grad():
            # Create dummy inputs
            batch_size = 2
            dummy_images = torch.randn(batch_size, 3, 224, 224).half().to(device)
            
            # Create dummy text inputs
            dummy_text = "Segment the red object <SEG>"
            inputs = model.tokenizer(
                [dummy_text] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            
            # Forward pass
            outputs = model(
                images=dummy_images,
                input_ids=inputs["input_ids"],
                multimask_output=False,
            )
            
            logger.info(f"Output keys: {list(outputs.keys())}")
            if "logits" in outputs:
                logger.info(f"Logits shape: {outputs['logits'].shape}")
            if "masks" in outputs:
                logger.info(f"Masks shape: {outputs['masks'].shape}")
            
        logger.info("✓ Forward pass successful")
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation
    logger.info("Testing generation...")
    try:
        with torch.no_grad():
            # Single image
            dummy_image = torch.randn(1, 3, 224, 224).half().to(device)
            prompt = "What is in this image? Segment the main object <SEG>"
            
            results = model.generate(
                images=dummy_image,
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.7,
                return_masks=True,
            )
            
            logger.info(f"Generated text: {results['text']}")
            if "masks" in results:
                logger.info(f"Generated {len(results['masks'])} masks")
            
        logger.info("✓ Generation successful")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("✓ All integration tests passed!")
    return True


if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)