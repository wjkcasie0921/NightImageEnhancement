import torch
import torch.nn as nn
import torch.nn.functional as F


def _select_gn_groups(num_channels: int, preferred_groups: int = 8) -> int:
    """
    Select a valid GroupNorm group count that divides num_channels.
    Falls back to 1 (LayerNorm-like) when needed.
    """
    groups = min(preferred_groups, num_channels)
    while groups > 1:
        if num_channels % groups == 0:
            return groups
        groups -= 1
    return 1


class ConvGNAct(nn.Module):
    """Conv + GroupNorm + SiLU, more stable than BatchNorm for small batch training."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        gn_groups = _select_gn_groups(out_channels, groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GhostBottleneckLite(nn.Module):
    """Lightweight Ghost-like block with depthwise cheap branch + residual."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 2)
        self.primary = ConvGNAct(channels, hidden, kernel_size=1)
        self.cheap = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(_select_gn_groups(hidden, 8), hidden),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Conv2d(hidden * 2, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_select_gn_groups(channels, 8), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.primary(x)
        c = self.cheap(p)
        out = torch.cat([p, c], dim=1)
        out = self.project(out)
        out = self.norm(out)
        return F.silu(out + x)


class LightweightBackbone(nn.Module):
    """Lightweight feature backbone for Retinex decomposition."""

    def __init__(self, width: int = 24):
        super().__init__()
        self.stem = ConvGNAct(3, width, kernel_size=3)
        self.block1 = GhostBottleneckLite(width)
        self.block2 = GhostBottleneckLite(width)
        self.block3 = GhostBottleneckLite(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class RetinexNetLite(nn.Module):
    """
    Retinex decomposition network with explicit reflectance prior:
    1) Predict smooth illumination I.
    2) Compute base reflectance R_base = x / I.
    3) Learn a tiny residual on reflectance to preserve details and avoid "oil painting".
    4) Use reflectance as enhanced output proxy during supervised training/eval.
    """

    def __init__(self, width: int = 24, illum_max: float = 1.2):
        super().__init__()
        self.backbone = LightweightBackbone(width=width)

        self.illum_head = nn.Sequential(
            ConvGNAct(width, width, kernel_size=3),
            nn.Conv2d(width, 1, kernel_size=1, bias=True),
        )

        # Input channels: feature (width) + low image (3) + base reflectance (3)
        self.refl_refine = nn.Sequential(
            ConvGNAct(width + 6, width, kernel_size=3),
            nn.Conv2d(width, 3, kernel_size=1, bias=True),
            nn.Tanh(),
        )

        self.eps = 1e-4
        self.illum_max = illum_max

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)

        # Keep illumination in [0.05, illum_max] to avoid unstable division and halos.
        illum = torch.sigmoid(self.illum_head(feat)) * (self.illum_max - 0.05) + 0.05

        # Explicit Retinex prior: reflectance approximates x / illum.
        illum_rgb = illum.repeat(1, 3, 1, 1)
        base_refl = (x / (illum_rgb + self.eps)).clamp(0.0, 1.0)

        # Tiny residual refinement keeps details while preventing over-smoothed textures.
        refl_res = self.refl_refine(torch.cat([feat, x, base_refl], dim=1)) * 0.1
        refl = (base_refl + refl_res).clamp(0.0, 1.0)
        return illum, refl

    @staticmethod
    def compose(illum: torch.Tensor, refl: torch.Tensor) -> torch.Tensor:
        """
        Build enhanced image from decomposition.
        For low-light enhancement, input x ~= R * I_low, so the bright target is
        better approximated by reflectance R than by re-multiplying low illumination.
        """
        _ = illum
        return refl.clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

