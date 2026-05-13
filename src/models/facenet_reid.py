import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from pathlib import Path

class FaceNetReID(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        
        # Look for local weights first (Offline cluster mode)
        local_weights_path = Path(__file__).resolve().parent.parent.parent / 'models_weights' / '20180402-114759-vggface2.pt'
        
        if local_weights_path.exists():
            # Initialize without downloading, then load weights manually
            self.backbone = InceptionResnetV1(pretrained=None, classify=False)
            state_dict = torch.load(local_weights_path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state_dict)
            print(f"Loaded Offline FaceNet Weights from: {local_weights_path}")
        else:
            # Fallback to online download
            self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
            
        self.embed_dim = embedding_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x is expected to be (B, 3, 160, 160)
        embeddings = self.backbone(x)
        logits = self.fc(self.dropout(embeddings))

        if self.training:
            return logits, embeddings
        else:
            return embeddings
