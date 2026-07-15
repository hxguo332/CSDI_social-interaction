import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def _set_requires_grad(module: nn.Module, flag: bool) -> None: # 控制整个模块的所有参数在训练时是否被更新
    for p in module.parameters(): # 遍历模块中的所有参数（权重和偏置）
        p.requires_grad = flag # 设置每个参数的梯度计算开关


class DoubleConv(nn.Module):
    """
    A light UNet-style conv block: (conv -> BN -> SiLU) x 2.
    Keeps it lightweight for small datasets.
    """

    def __init__(self, in_ch: int, out_ch: int):  # PyTorch 图像格式是 (B, C, H, W)
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_ch), # 批归一化，稳定训练，加速收敛
            nn.SiLU(inplace=True), 
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 类型提示：返回值是一个张量
        return self.block(x)


class UpBlock(nn.Module):
    """
    UNet upsampling block: upsample -> concat skip -> double conv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # Pad if shapes mismatch by 1 due to odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18MapEncoder(nn.Module):
    """
    UNet-style decoder on top of a ResNet-18 encoder.

    - Encoder: torchvision ResNet-18 (ImageNet weights by default)
      Features used (with input size HxW):
        c1: after stem (H/2, 64)
        c2: after layer1 (H/4, 64)
        c3: after layer2 (H/8, 128)
        c4: after layer3 (H/16, 256)
        c5: after layer4 (H/32, 512)

    - Decoder: upsample c5 -> c4 -> c3 -> c2 -> c1, producing relatively
      high-resolution feature maps suitable for ROI Align.

    Outputs
    -------
    If return_pyramid=True, returns a dict with keys:
      - 'p2': stride 4 map  (from d3)
      - 'p3': stride 8 map  (from d4)
      - 'p4': stride 16 map (from d5)
      - 'p5': stride 32 map (1x1 reduced c5)

    Otherwise, returns a single high-res map at stride 4 (p2).

    Notes
    -----
    - Designed to be light: decoder channels kept modest.
    - Supports freezing early encoder stages via `finetune_from`.
    - If input channels != 3, adapts the first conv appropriately.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 128,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        finetune_from: str | None = "layer4",  # one of: None, 'all', 'conv1','bn1','layer1','layer2','layer3','layer4'
        return_pyramid: bool = True,
    ) -> None:
        super().__init__()

        # Build encoder (ResNet18)
        resnet = models.resnet18(weights=weights)

        # If input channels differ, adapt conv1 while keeping weights if reasonable
        if in_channels != 3:
            old_conv1 = resnet.conv1
            new_conv1 = nn.Conv2d(
                in_channels, old_conv1.out_channels, kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride, padding=old_conv1.padding, bias=False,
            )
            with torch.no_grad():
                if in_channels == 1:
                    # Average RGB weights to single channel
                    w = old_conv1.weight.sum(dim=1, keepdim=True) / 3.0
                    new_conv1.weight.copy_(w)
                elif in_channels > 3:
                    # Copy first 3 channels; zero-init extras
                    new_conv1.weight.zero_()
                    new_conv1.weight[:, :3].copy_(old_conv1.weight)
                else:  # in_channels == 2
                    new_conv1.weight.zero_()
                    new_conv1.weight[:, :2].copy_(old_conv1.weight[:, :2])
            resnet.conv1 = new_conv1

        # Expose encoder stages
        self.enc_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # H/2, 64
        self.enc_pool = resnet.maxpool  # H/4
        self.enc_layer1 = resnet.layer1  # H/4, 64
        self.enc_layer2 = resnet.layer2  # H/8, 128
        self.enc_layer3 = resnet.layer3  # H/16, 256
        self.enc_layer4 = resnet.layer4  # H/32, 512

        # Optional fine-tuning control
        stages = {
            "conv1": self.enc_conv1,
            "bn1": None,  # folded into enc_conv1
            "layer1": self.enc_layer1,
            "layer2": self.enc_layer2,
            "layer3": self.enc_layer3,
            "layer4": self.enc_layer4,
        }
        order = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        self._frozen_modules: list[nn.Module] = []
        if finetune_from is None:
            for name in order:
                mod = stages[name]
                if mod is not None:
                    _set_requires_grad(mod, False)
                    self._frozen_modules.append(mod)
        elif finetune_from == "all":
            for name in order:
                mod = stages[name]
                if mod is not None:
                    _set_requires_grad(mod, True)
        else:
            if finetune_from not in order:
                raise ValueError(f"finetune_from must be one of {order + ['all', None]} but got {finetune_from}")
            start_idx = order.index(finetune_from)
            for i, name in enumerate(order):
                mod = stages[name]
                if mod is None:
                    continue
                trainable = i >= start_idx
                _set_requires_grad(mod, trainable)
                if not trainable:
                    self._frozen_modules.append(mod)

        # Lateral 1x1 to unify channels before decoder taps (keeps decoder light)
        self.lateral_c5 = nn.Conv2d(512, feat_channels, 1, bias=False)
        self.lateral_c4 = nn.Conv2d(256, feat_channels, 1, bias=False)
        self.lateral_c3 = nn.Conv2d(128, feat_channels, 1, bias=False)
        self.lateral_c2 = nn.Conv2d(64, feat_channels, 1, bias=False)
        self.lateral_c1 = nn.Conv2d(64, feat_channels, 1, bias=False)

        # Decoder blocks (UNet-style up + concat + conv)
        self.up_c5_c4 = UpBlock(feat_channels, feat_channels, feat_channels)  # -> H/16
        self.up_c4_c3 = UpBlock(feat_channels, feat_channels, feat_channels)  # -> H/8
        self.up_c3_c2 = UpBlock(feat_channels, feat_channels, feat_channels)  # -> H/4
        # Optional further up to H/2 if needed later
        self.up_c2_c1 = UpBlock(feat_channels, feat_channels, feat_channels)  # -> H/2

        # Output projection heads for pyramid maps
        self.out_p5 = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)  # stride 32
        self.out_p4 = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)  # stride 16
        self.out_p3 = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)  # stride 8
        self.out_p2 = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)  # stride 4

        self.return_pyramid = return_pyramid

    def extract_encoder_feats(self, x: torch.Tensor):
        # Stem
        c1 = self.enc_conv1(x)          # H/2, 64
        x = self.enc_pool(c1)           # H/4
        c2 = self.enc_layer1(x)         # H/4, 64
        c3 = self.enc_layer2(c2)        # H/8, 128
        c4 = self.enc_layer3(c3)        # H/16, 256
        c5 = self.enc_layer4(c4)        # H/32, 512
        return c1, c2, c3, c4, c5

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Returns
        -------
        - If return_pyramid: dict with keys 'p2','p3','p4','p5' corresponding
          to stride 4/8/16/32 feature maps (all with `feat_channels` channels).
        - Else: single high-resolution feature map at stride 4 (Tensor).
        """
        c1, c2, c3, c4, c5 = self.extract_encoder_feats(x)

        # Lateral projections
        l5 = self.lateral_c5(c5)  # H/32
        l4 = self.lateral_c4(c4)  # H/16
        l3 = self.lateral_c3(c3)  # H/8
        l2 = self.lateral_c2(c2)  # H/4
        l1 = self.lateral_c1(c1)  # H/2

        # Decoder path (UNet-style)
        d5 = l5                                # H/32
        d4 = self.up_c5_c4(d5, l4)             # H/16
        d3 = self.up_c4_c3(d4, l3)             # H/8
        d2 = self.up_c3_c2(d3, l2)             # H/4
        # Optionally continue one more step if needed later:
        # d1 = self.up_c2_c1(d2, l1)           # H/2

        p5 = self.out_p5(d5)
        p4 = self.out_p4(d4)
        p3 = self.out_p3(d3)
        p2 = self.out_p2(d2)

        if self.return_pyramid:
            return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
        else:
            return p2  # stride-4 high-resolution map by default

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen modules' BNs in eval to preserve pre-trained stats
        for m in getattr(self, "_frozen_modules", []):
            m.eval()
        return self


if __name__ == "__main__":
    # Small smoke test
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = UNetResNet18MapEncoder(
        in_channels=3,
        feat_channels=128,
        weights=ResNet18_Weights.DEFAULT,
        finetune_from="layer4",
        return_pyramid=True,
    ).to(device)

    #x = torch.randn(2, 3, 512, 2048).to(device)
    x = torch.randn(2, 3, 512, 512).to(device)
    out = model(x)
    assert isinstance(out, dict)
    print({k: v.shape for k, v in out.items()})

