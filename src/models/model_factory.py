import torch.nn as nn
import torchvision.models as models

class ReidBaseline(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            resnet = models.resnet50(weights=None)
        except TypeError:
            # Fallback for older torchvision versions on clusters like DelftBlue
            resnet = models.resnet50(pretrained=False)
        # 1-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.embed_dim = resnet.fc.in_features
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return features

def build_reid_model(backbone_name: str, num_classes: int, **kwargs):
    if backbone_name == "resnet50":
        return ReidBaseline(num_classes)
    elif backbone_name == "facenet":
        try:
            import facenet_pytorch
        except ImportError:
            raise ImportError("facenet_pytorch is not installed. Please run: pip install facenet-pytorch")

        from src.models.facenet_reid import FaceNetReID
        return FaceNetReID(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
