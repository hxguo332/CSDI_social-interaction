import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResnetMapEncoder(nn.Module):
    def __init__(
        self,
        output_dim=256,
        weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
        grid_size: int | tuple[int, int] = 7,
        finetune_from: str | None = "layer4",
    ):
        super(ResnetMapEncoder, self).__init__()
        # Normalize grid_size to tuple
        if isinstance(grid_size, int):
            grid_h, grid_w = grid_size, grid_size
        else:
            grid_h, grid_w = grid_size

        # Load ResNet-18 with explicit weights enum (pretrained deprecates)
        resnet = models.resnet18(weights=weights)

        # Optional light fine-tuning: unfreeze from a given stage (default: layer4)
        self._frozen_modules = []
        self.finetune_from = finetune_from
        stages = {
            "conv1": resnet.conv1,
            "bn1": resnet.bn1,
            "layer1": resnet.layer1,
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        }
        order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

        def set_requires_grad(module: nn.Module, flag: bool):
            for p in module.parameters():
                p.requires_grad = flag

        if finetune_from is None:
            # Freeze all backbone layers
            for name in order:
                set_requires_grad(stages[name], False)
                self._frozen_modules.append(stages[name])
        elif finetune_from == "all":
            # Train all layers
            for name in order:
                set_requires_grad(stages[name], True)
        else:
            if finetune_from not in order:
                raise ValueError(f"finetune_from must be one of {order + ['all', None]} but got {finetune_from}")
            # Freeze everything before finetune_from
            start_idx = order.index(finetune_from)
            for i, name in enumerate(order):
                trainable = i >= start_idx
                set_requires_grad(stages[name], trainable)
                if not trainable:
                    self._frozen_modules.append(stages[name])

        # Keep only convolutional trunk (no avgpool/fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Adaptive pooling to a fixed grid to retain spatial layout
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
        self._feat_dim = 512 * grid_h * grid_w
        self.fc = nn.Linear(self._feat_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # safe for non-contiguous tensors
        x = self.fc(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen modules' BNs in eval to preserve pre-trained stats
        for m in self._frozen_modules:
            m.eval()
        return self

if __name__ =="__main__":
    # Instantiate model
    map_encoder = ResnetMapEncoder(output_dim=256, weights=ResNet18_Weights.DEFAULT, grid_size=7, finetune_from="layer4")

    # Test with a random input (1 grayscale map of size 500x2000)
    sample_input = torch.randn(2, 3, 500, 2000)  # (batch, channels, height, width)
    output_embedding = map_encoder(sample_input)
    print("Output Shape:", output_embedding.shape)  # Expected: (1, 256)
