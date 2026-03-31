import torch.nn as nn
import torch
from ..compat import ensure_torch_xpu_compat

ensure_torch_xpu_compat()

from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from .attention import CubeDiffTransformerBlock
from .norm import CubeDiffGroupNorm


def freeze(module: nn.Module) -> None:
    """
        Freeze all parameters in a module so they do not require gradients.
    """
    for p in module.parameters():
        p.requires_grad = False

def swap_transformer_blocks(root: nn.Module) -> None:
    """Replace every `BasicTransformerBlock` inside `Transformer2DModel`."""
    for child in root.children():
        swap_transformer_blocks(child)
        if isinstance(child, Transformer2DModel):
            for i, blk in enumerate(child.transformer_blocks):
                if isinstance(blk, BasicTransformerBlock):
                    new_blk = CubeDiffTransformerBlock(
                        dim=blk.dim,
                        num_attention_heads=blk.num_attention_heads,
                        attention_head_dim=blk.attention_head_dim,
                        dropout=blk.dropout,
                        cross_attention_dim=blk.cross_attention_dim,
                        activation_fn=blk.activation_fn,
                        num_embeds_ada_norm=getattr(child.config, 'num_embeds_ada_norm', None),
                        attention_bias=blk.attention_bias,
                        only_cross_attention=blk.only_cross_attention,
                        double_self_attention=blk.double_self_attention,
                        norm_elementwise_affine=blk.norm_elementwise_affine,
                        norm_type=blk.norm_type,
                        norm_eps=getattr(child.config, 'norm_eps', 1e-5),
                        upcast_attention=getattr(child.config, 'upcast_attention', False),
                        attention_type=getattr(child.config, 'attention_type', 'default'),
                    )
                    # Load the state dict with proper error handling
                    try:
                        new_blk.load_state_dict(blk.state_dict(), strict=False)
                    except RuntimeError as e:
                        print(f"Warning: Could not load state dict completely: {e}")
                        # Copy compatible weights manually
                        new_state = new_blk.state_dict()
                        old_state = blk.state_dict()
                        for key in new_state.keys():
                            if key in old_state and new_state[key].shape == old_state[key].shape:
                                new_state[key].copy_(old_state[key])
                        new_blk.load_state_dict(new_state)
                    
                    child.transformer_blocks[i] = new_blk


def load_sliced_unet_weights(unet: UNet2DConditionModel, state_dict: dict) -> None:
    """
    Load a CubeDiff state_dict (which may have 7-channel conv_in) into a 4-channel UNet.
    
    Performs "weight surgery": slices conv_in.weight from [320, 7, 3, 3] to [320, 4, 3, 3]
    so that the pretrained CubeDiff weights can be safely loaded into a standard 4-channel UNet.
    
    Args:
        unet: The target UNet2DConditionModel with 4-channel conv_in.
        state_dict: The source state_dict, potentially from a 7-channel CubeDiff checkpoint.
    """
    sliced_keys = []
    for key in list(state_dict.keys()):
        if key.endswith("conv_in.weight"):
            tensor = state_dict[key]
            if tensor.shape[1] == 7:
                state_dict[key] = tensor[:, :4, :, :]
                sliced_keys.append(key)
                print(f"[CubeDiff] Sliced {key} from {list(tensor.shape)} to {list(state_dict[key].shape)}")
    
    # Load with strict=False to skip any remaining mismatched keys
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    
    if sliced_keys:
        print(f"[CubeDiff] Successfully sliced conv_in.weight from 7 to 4 channels.")
    if missing:
        print(f"[CubeDiff] Missing keys (expected): {len(missing)} keys")
    if unexpected:
        print(f"[CubeDiff] Unexpected keys (ignored): {len(unexpected)} keys")


def expand_unet_conv_in(unet: UNet2DConditionModel, in_channels: int = 7) -> None:
    """
    Replace UNet conv_in so the network accepts CubeDiff's 7-channel latent input.
    """
    old_conv = unet.conv_in
    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        copy_channels = min(old_conv.in_channels, in_channels)
        new_conv.weight[:, :copy_channels].copy_(old_conv.weight[:, :copy_channels])
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv


def patch_unet(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    """Patch a base UNet to CubeDiff architecture (attention blocks only, no channel expansion)."""

    # Swap transformer blocks
    swap_transformer_blocks(unet)
    
    return unet


def patch_groupnorm(root: nn.Module, num_faces: int = 6) -> None:
    """Recursively replace GroupNorm with CubeDiffGroupNorm (in-place)."""
    for name, child in list(root.named_children()):
        patch_groupnorm(child, num_faces=num_faces)
        if isinstance(child, nn.GroupNorm) and not isinstance(child, CubeDiffGroupNorm):
            setattr(root, name, CubeDiffGroupNorm(child, num_faces=num_faces))
