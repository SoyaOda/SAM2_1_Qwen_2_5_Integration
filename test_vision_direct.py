"""
Direct test for vision encoder compatibility.
"""

import torch
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.model_config import VisionEncoderConfig
from models.vision_encoder import VisionEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_vision_encoder_shapes():
    """Test vision encoder with different configurations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test 1: Small configuration that works in test_basic.py
    logger.info("Test 1: Small configuration (224x224)")
    config1 = VisionEncoderConfig(
        image_size=224,
        patch_size=14,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
    )
    
    encoder1 = VisionEncoder(config1).to(device)
    dummy_input1 = torch.randn(2, 3, 224, 224).to(device)
    
    try:
        output1 = encoder1(dummy_input1, return_features=True)
        logger.info(f"✓ Output shape: {output1.shape}")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    # Test 2: Original configuration
    logger.info("\nTest 2: Original configuration (896x896)")
    config2 = VisionEncoderConfig(
        image_size=896,
        patch_size=14,
        hidden_dim=1280,
        num_layers=32,
        num_heads=20,
    )
    
    # Check if dimensions are compatible
    logger.info(f"Hidden dim: {config2.hidden_dim}")
    logger.info(f"Num heads: {config2.num_heads}")
    logger.info(f"Head dim: {config2.hidden_dim // config2.num_heads}")
    
    encoder2 = VisionEncoder(config2).to(device)
    dummy_input2 = torch.randn(1, 3, 896, 896).to(device)
    
    try:
        output2 = encoder2(dummy_input2, return_features=True)
        logger.info(f"✓ Output shape: {output2.shape}")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Modified configuration with compatible dimensions
    logger.info("\nTest 3: Modified configuration (896x896, adjusted heads)")
    config3 = VisionEncoderConfig(
        image_size=896,
        patch_size=14,
        hidden_dim=1280,
        num_layers=32,
        num_heads=16,  # 1280/16 = 80
    )
    
    encoder3 = VisionEncoder(config3).to(device)
    dummy_input3 = torch.randn(1, 3, 896, 896).to(device)
    
    try:
        output3 = encoder3(dummy_input3, return_features=True)
        logger.info(f"✓ Output shape: {output3.shape}")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")


if __name__ == "__main__":
    test_vision_encoder_shapes()