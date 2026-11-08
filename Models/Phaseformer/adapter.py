"""Load the official Phaseformer UIEB model safely and adapt its two-scale restoration output to the pipeline's 256-pixel RGB inputs."""
from __future__ import annotations

CHECKPOINT_FILENAME = "best.pth"
DEPENDENCIES = ()


def load_checkpoint(torch, path, device):
    """Checkpoint path and device -> tensor state dictionary from the official model."""
    import model_with_eca as architecture

    allowed = [getattr(architecture, name) for name in (
        "Restormer", "ECA", "MDTA", "GDFN", "TransformerBlock", "DownSample", "UpSample", "UpSample1",
    )]
    allowed += [getattr(torch.nn, name) for name in (
        "Conv1d", "Conv2d", "Sequential", "ModuleList", "Sigmoid", "AdaptiveAvgPool2d",
        "PixelShuffle", "PixelUnshuffle", "LayerNorm",
    )]
    with torch.serialization.safe_globals(allowed):
        saved = torch.load(path, map_location=device, weights_only=True)
    return saved.state_dict() if isinstance(saved, torch.nn.Module) else saved


def build_model(torch, device, state):
    """Tensor state dictionary and device -> strictly loaded Phaseformer model."""
    from model_with_eca import Restormer

    model = Restormer().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> restored RGB tensors at the input dimensions."""
    prediction = model(tensor * 2.0 - 1.0)[1]
    prediction = ((prediction + 1.0) / 2.0).clamp(0, 1)
    return torch.nn.functional.interpolate(
        prediction, size=tensor.shape[-2:], mode="bicubic", align_corners=False,
    ).clamp(0, 1)
