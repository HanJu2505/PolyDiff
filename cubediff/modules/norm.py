import torch.nn as nn

class CubeDiffGroupNorm(nn.Module):
    def __init__(self, original_norm: nn.GroupNorm, num_faces: int = 6, sync_enabled: bool = True):
        super().__init__()
        self.num_faces = num_faces
        self.sync_enabled = sync_enabled

        # Create internal GroupNorm with same config
        self.norm = nn.GroupNorm(
            num_groups=original_norm.num_groups,
            num_channels=original_norm.num_channels,
            eps=original_norm.eps,
            affine=original_norm.affine
        )

        # Copy original weights/bias
        if original_norm.affine:
            self.norm.weight.data.copy_(original_norm.weight.data)
            self.norm.bias.data.copy_(original_norm.bias.data)

    def forward(self, x):
        """
        x: (B*T, C, H, W) or (B*T, C, H*W) → Cube-aware reshape → Apply shared GroupNorm → reshape back
        """

        if not self.sync_enabled or x.shape[0] < self.num_faces:
            # fallback: standard groupnorm without reshaping
            return self.norm(x)

        if len(x.shape) == 4: #(BT, C, H, W)
            bt, c, h, w = x.shape
            T = self.num_faces
            B = bt // T
            assert bt == B * T, f"Input batch size {bt} is not divisible by num_faces {T}"

            # Reshape across cube faces
            x = x.reshape(B, T, c, h, w).permute(0, 2, 1, 3, 4).reshape(B, c, T * h * w)

            # Apply GroupNorm across combined spatial area
            x = self.norm(x)

            # Reshape back
            x = x.reshape(B, c, T, h, w).permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)

        elif len(x.shape) == 3: # (BT, C, HW)
            bt, c, hw = x.shape
            T = self.num_faces
            B = bt // T
            assert bt == B * T, f"Input batch size {bt} is not divisible by num_faces {T}"

            # Reshape across cube faces
            x = x.reshape(B, T, c, hw).permute(0, 2, 1, 3).reshape(B, c, T * hw)
            # Apply GroupNorm across combined spatial area
            x = self.norm(x)
            # Reshape back
            x = x.reshape(B, c, T, hw).permute(0, 2, 1, 3).reshape(bt, c, hw)
        
        return x
