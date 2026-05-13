import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FaceNetReID(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, dropout: float = 0.0):
        super().__init__()
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

