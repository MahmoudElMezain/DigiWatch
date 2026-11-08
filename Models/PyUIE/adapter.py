"""Restore RGB images with the official PyUIE pyramid network using either its UIEB or LSUI tensor checkpoint."""
from __future__ import annotations

CHECKPOINT_FILENAME = "UIEB.pt"
DEPENDENCIES = ()


def build_model(torch, device, state):
    """Tensor state dictionary and device -> strictly loaded PyUIE model."""
    from PyramidNet import LPTN

    model = LPTN().to(device)
    state = dict(state)
    for key in list(state):
        if key.startswith("trans_high.DCSEB2."):
            original = key.replace("trans_high.DCSEB2.", "trans_high.DCSEB2_2.", 1)
            if original not in state or not torch.equal(state[key], state[original]):
                raise ValueError(f"PyUIE checkpoint contains inconsistent duplicate parameter: {key}")
            del state[key]
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> full-resolution restored RGB tensors."""
    red, green, blue = tensor[:, 2], tensor[:, 1], tensor[:, 0]
    red_mean = red.mean(dim=(1, 2), keepdim=True) / 256.0
    green_mean = green.mean(dim=(1, 2), keepdim=True) / 256.0
    blue_mean = blue.mean(dim=(1, 2), keepdim=True) / 256.0
    compensated = torch.stack((
        blue + (green_mean - blue_mean) * (1 - blue_mean) * green,
        green,
        red + (green_mean - red_mean) * (1 - red_mean) * green,
    ), dim=1).clamp(0, 1)
    averages = compensated.mean(dim=(2, 3), keepdim=True)
    batch_size, channels, height, width = compensated.shape
    prepared = torch.zeros_like(compensated)
    for channel in range(channels):
        total = compensated[:, channel].sum(dim=(1, 2), keepdim=True)
        scale = torch.where(total > 0, height * width / total, torch.zeros_like(total))
        weight = (averages[:, channel] * scale).expand(batch_size, height, width)
        prepared[:, channel] = compensated[:, channel] * weight
    prepared = prepared.clamp(0, 1)
    return model(prepared)[1].clamp(0, 1)
