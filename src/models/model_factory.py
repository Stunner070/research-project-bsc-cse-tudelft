def build_reid_model(backbone_name: str, num_classes: int, **kwargs):
    if backbone_name == "resnet50":
        from src.scripts.train_compare_representations import ReidBaseline
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

