"""
LLM Integration module for LISA-改.

This module handles the integration between the vision encoder,
mask decoder, and Qwen2.5 LLM for multimodal understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType
import logging

from models.tokenizer_extension import ExtendedQwenTokenizer
from configs.model_config import LLMIntegrationConfig

logger = logging.getLogger(__name__)


class LLMProjector(nn.Module):
    """
    Projects between different embedding dimensions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "gelu"
    ):
        super().__init__()
        
        if num_layers == 1:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            hidden_dim = (input_dim + output_dim) // 2
            
            # First layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
            ])
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True),
                ])
            
            # Final layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SegmentationTokenHandler(nn.Module):
    """
    Handles special segmentation tokens in the LLM output.
    """
    
    def __init__(
        self,
        llm_dim: int = 2048,
        mask_decoder_dim: int = 512,
        seg_token_id: int = -1,
    ):
        super().__init__()
        self.llm_dim = llm_dim
        self.mask_decoder_dim = mask_decoder_dim
        self.seg_token_id = seg_token_id
        
        # Projection from LLM hidden state to mask decoder prompt
        self.seg_projector = LLMProjector(
            llm_dim,
            mask_decoder_dim,
            num_layers=2
        )
    
    def extract_seg_embeddings(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings at <SEG> token positions.
        
        Args:
            hidden_states: LLM hidden states (B, seq_len, hidden_dim)
            input_ids: Token IDs (B, seq_len)
            
        Returns:
            seg_embeddings: Embeddings at <SEG> positions (num_segs, hidden_dim)
            seg_positions: Positions of <SEG> tokens (num_segs, 2) as (batch_idx, seq_idx)
        """
        # Find <SEG> token positions
        seg_mask = (input_ids == self.seg_token_id)
        seg_positions = torch.nonzero(seg_mask, as_tuple=False)
        
        if seg_positions.shape[0] == 0:
            return None, None
        
        # Extract embeddings at these positions
        batch_indices = seg_positions[:, 0]
        seq_indices = seg_positions[:, 1]
        seg_embeddings = hidden_states[batch_indices, seq_indices]
        
        return seg_embeddings, seg_positions
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process <SEG> tokens and return mask decoder prompts.
        """
        seg_embeddings, seg_positions = self.extract_seg_embeddings(
            hidden_states, input_ids
        )
        
        if seg_embeddings is None:
            return None, None
        
        # Project to mask decoder dimension
        mask_prompts = self.seg_projector(seg_embeddings)
        
        return mask_prompts, seg_positions


class VisionSummaryInjector(nn.Module):
    """
    Injects vision summary tokens into LLM input.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        llm_dim: int = 2048,
        num_tokens: int = 1,
        vis_sum_token_id: int = -1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.vis_sum_token_id = vis_sum_token_id
        
        # Project from vision to LLM dimension
        self.vision_projector = LLMProjector(
            vision_dim,
            llm_dim * num_tokens,
            num_layers=2
        )
    
    def forward(
        self,
        vision_summary: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject vision summary at <VIS_SUM> token positions.
        
        Args:
            vision_summary: Vision summary features (B, vision_dim)
            input_embeds: LLM input embeddings (B, seq_len, llm_dim)
            input_ids: Token IDs (B, seq_len)
            
        Returns:
            Modified input embeddings
        """
        B, seq_len, llm_dim = input_embeds.shape
        
        # Project vision features
        vision_tokens = self.vision_projector(vision_summary)
        vision_tokens = vision_tokens.view(B, self.num_tokens, llm_dim)
        
        # Find <VIS_SUM> positions
        vis_sum_mask = (input_ids == self.vis_sum_token_id)
        
        # Replace embeddings at these positions
        for b in range(B):
            positions = torch.nonzero(vis_sum_mask[b], as_tuple=False).squeeze(-1)
            if positions.numel() > 0:
                # Take first position if multiple
                pos = positions[0].item()
                # Replace with vision token(s)
                for i in range(min(self.num_tokens, seq_len - pos)):
                    input_embeds[b, pos + i] = vision_tokens[b, i]
        
        return input_embeds


class LLMIntegration(nn.Module):
    """
    Main LLM integration module for LISA-改.
    
    Handles:
    - Loading and configuring Qwen2.5 LLM
    - LoRA adaptation
    - Special token processing
    - Vision-language alignment
    """
    
    def __init__(
        self,
        config: LLMIntegrationConfig,
        tokenizer: ExtendedQwenTokenizer,
        model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Load base LLM
        logger.info(f"Loading LLM from {model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Resize embeddings for new tokens
        self.llm = tokenizer.resize_model_embeddings(self.llm)
        
        # Apply LoRA if configured
        if config.use_lora:
            self._apply_lora()
        
        # Special token IDs
        self.seg_token_id = tokenizer.get_seg_token_id()
        self.vis_sum_token_id = tokenizer.get_vis_sum_token_id()
        self.depth_token_id = tokenizer.get_depth_token_id()
        self.depth_sum_token_id = tokenizer.tokenizer.convert_tokens_to_ids("<DEPTH_SUM>")
        
        # Token handlers
        self.seg_handler = SegmentationTokenHandler(
            llm_dim=config.llm_hidden_dim,
            mask_decoder_dim=config.mask_token_proj_dim,
            seg_token_id=self.seg_token_id,
        )
        
        self.vision_injector = VisionSummaryInjector(
            vision_dim=config.mask_token_proj_dim,
            llm_dim=config.llm_hidden_dim,
            vis_sum_token_id=self.vis_sum_token_id,
        )
        
        # Depth token injector (for future v1.5)
        self.depth_injector = None
        if config.depth_head and config.depth_head.use_depth_tokens:
            self.depth_injector = VisionSummaryInjector(
                vision_dim=config.mask_token_proj_dim,
                llm_dim=config.llm_hidden_dim,
                vis_sum_token_id=self.depth_sum_token_id,
            )
        
        logger.info("LLM integration initialized")
    
    def _apply_lora(self):
        """Apply LoRA adaptation to the LLM."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        logger.info(f"Applied LoRA with rank {self.config.lora_rank}")
    
    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        vision_summary: Optional[torch.Tensor] = None,
        depth_summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for LLM forward pass.
        
        Args:
            input_ids: Token IDs (B, seq_len)
            vision_summary: Vision features to inject (B, vision_dim)
            depth_summary: Depth features to inject (B, depth_dim)
            
        Returns:
            Dict with prepared inputs
        """
        # Get base embeddings
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Inject vision summary if available
        if vision_summary is not None:
            input_embeds = self.vision_injector(
                vision_summary, input_embeds, input_ids
            )
        
        # Inject depth summary if available and configured
        if depth_summary is not None and self.depth_injector is not None:
            input_embeds = self.depth_injector(
                depth_summary, input_embeds, input_ids
            )
        
        return {
            "inputs_embeds": input_embeds,
            "attention_mask": torch.ones_like(input_ids),
        }
    
    def extract_seg_prompts(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract segmentation prompts from LLM outputs.
        
        Args:
            outputs: LLM outputs
            input_ids: Input token IDs
            
        Returns:
            seg_prompts: Segmentation prompts for mask decoder
            seg_positions: Positions of <SEG> tokens
        """
        # Get hidden states from last layer
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            # Fallback: use output logits and project back
            hidden_states = self.llm.lm_head.weight[outputs.logits.argmax(-1)]
        
        # Extract segmentation prompts
        seg_prompts, seg_positions = self.seg_handler(hidden_states, input_ids)
        
        return seg_prompts, seg_positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        vision_summary: Optional[torch.Tensor] = None,
        depth_summary: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_seg_prompts: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LLM with multimodal integration.
        
        Args:
            input_ids: Token IDs
            vision_summary: Vision features
            depth_summary: Depth features
            labels: Target labels for training
            return_seg_prompts: Whether to extract segmentation prompts
            
        Returns:
            Dict containing:
                - logits: LLM output logits
                - loss: Language modeling loss (if labels provided)
                - seg_prompts: Segmentation prompts (if requested)
                - seg_positions: Positions of <SEG> tokens
        """
        # Prepare inputs
        model_inputs = self.prepare_inputs(
            input_ids, vision_summary, depth_summary
        )
        
        # Add labels if provided
        if labels is not None:
            model_inputs["labels"] = labels
        
        # Forward through LLM
        outputs = self.llm(
            **model_inputs,
            output_hidden_states=return_seg_prompts,
            **kwargs,
        )
        
        results = {
            "logits": outputs.logits,
        }
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            results["loss"] = outputs.loss
        
        # Extract segmentation prompts if requested
        if return_seg_prompts:
            seg_prompts, seg_positions = self.extract_seg_prompts(
                outputs, input_ids
            )
            results["seg_prompts"] = seg_prompts
            results["seg_positions"] = seg_positions
        
        return results
    
    def generate(
        self,
        input_ids: torch.Tensor,
        vision_summary: Optional[torch.Tensor] = None,
        depth_summary: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Generate text with segmentation token detection.
        
        Returns:
            generated_ids: Generated token IDs
            seg_info: List of dicts with segmentation info
        """
        # Prepare inputs
        model_inputs = self.prepare_inputs(
            input_ids, vision_summary, depth_summary
        )
        
        # Track segmentation tokens during generation
        seg_info = []
        
        # Custom generation loop to detect <SEG> tokens
        # (Simplified version - in practice, use HuggingFace generate with callbacks)
        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        
        # Post-process to find <SEG> tokens
        output_ids = generated_ids.sequences
        for batch_idx in range(output_ids.shape[0]):
            seq = output_ids[batch_idx]
            seg_positions = (seq == self.seg_token_id).nonzero(as_tuple=False)
            
            for pos in seg_positions:
                seg_info.append({
                    "batch_idx": batch_idx,
                    "position": pos.item(),
                    "token_id": self.seg_token_id,
                })
        
        return output_ids, seg_info