import torch
import torch.nn as nn
import torchvision.models as models

class ResnetMapEncoder(nn.Module):
    def __init__(self, output_dim=256, pretrained=True):
        super(ResnetMapEncoder, self).__init__()
        # Load ResNet-18 and keep only convolutional layers (remove GAP & FC layers)
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove last two layers

        # Adaptive Pooling to get a fixed-size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))  # Ensures (batch, 512, 1, 1)
        self.fc = nn.Linear(512, output_dim)  # Reduce feature size to output_dim

    def forward(self, x):
        x = self.backbone(x)  # Extract features using ResNet-18 backbone
        x = self.adaptive_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # (batch, output_dim)
        return x  # Fixed-size feature vector

if __name__ =="__main__":
    # Instantiate model
    map_encoder = ResnetMapEncoder(output_dim=256)

    # Test with a random input (1 grayscale map of size 500x2000)
    sample_input = torch.randn(1, 3, 500, 2000)  # (batch, channels, height, width)
    output_embedding = map_encoder(sample_input)
    print("Output Shape:", output_embedding.shape)  # Expected: (1, 256)