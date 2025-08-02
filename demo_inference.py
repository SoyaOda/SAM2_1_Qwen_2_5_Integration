"""
Demo inference script for LISA-改.

This script demonstrates how to use the integrated model
for language-guided segmentation.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging
import argparse
from PIL import Image
import matplotlib.pyplot as plt

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


def load_image(image_path: str, target_size: int = 896) -> torch.Tensor:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # Normalize (ImageNet mean and std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor.unsqueeze(0)  # Add batch dimension


def visualize_results(
    image: torch.Tensor,
    masks: list,
    text: str,
    save_path: str = None
):
    """Visualize segmentation results."""
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.squeeze(0) * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).numpy()
    
    # Create figure
    num_masks = len(masks) if masks else 0
    fig, axes = plt.subplots(1, num_masks + 1, figsize=(5 * (num_masks + 1), 5))
    
    if num_masks == 0:
        axes = [axes]
    elif num_masks == 1:
        axes = [axes[0], axes[1]]
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show masks
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        
        # Apply threshold
        mask = (mask > 0).astype(np.float32)
        
        # Create colored overlay
        colored_mask = np.zeros((*mask.shape, 3))
        colored_mask[:, :, 0] = mask  # Red channel
        
        # Overlay on image
        overlay = image.copy()
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + colored_mask[mask > 0.5] * 0.5
        
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"Mask {i + 1}")
        axes[i + 1].axis('off')
    
    plt.suptitle(f"Prompt: {text}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def demo_inference(
    image_path: str,
    text_prompt: str,
    model_config: LISAConfig,
    device: str = "cuda",
    visualize: bool = True,
):
    """Run demo inference."""
    logger.info("Starting LISA-改 demo inference...")
    
    # Load image
    logger.info(f"Loading image: {image_path}")
    image = load_image(image_path, target_size=model_config.vision_encoder.image_size)
    image = image.to(device)
    
    # Create model
    logger.info("Initializing LISA-改 model...")
    try:
        model = LISAModel(model_config)
        model = model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        logger.error(
            "Note: This demo requires downloading pretrained models. "
            "Make sure you have internet connection and sufficient disk space."
        )
        return
    
    # Run inference
    logger.info(f"Running inference with prompt: '{text_prompt}'")
    try:
        with torch.no_grad():
            results = model.generate(
                images=image,
                prompt=text_prompt,
                max_new_tokens=128,
                temperature=0.7,
                return_masks=True,
            )
        
        logger.info(f"Generated text: {results['text']}")
        
        # Visualize results
        if visualize and 'masks' in results:
            visualize_results(
                image=image,
                masks=results['masks'],
                text=text_prompt,
                save_path=f"demo_output_{Path(image_path).stem}.png"
            )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LISA-改 Demo Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for segmentation"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="3B",
        choices=["3B", "7B", "72B"],
        help="Model size to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        return 1
    
    # Create model config
    config = LISAConfig(model_size=args.model_size)
    
    # Disable depth for v0
    if config.depth_head:
        config.depth_head.enable = False
    
    # Run demo
    try:
        demo_inference(
            image_path=args.image,
            text_prompt=args.prompt,
            model_config=config,
            device=args.device,
            visualize=not args.no_viz,
        )
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())