"""Load the official D2Mamba UIEB model and apply its A-star enhancement path to RGB tensors."""
import argparse

CHECKPOINT_FILENAME = "netG_UIEB.pt"
DEPENDENCIES = ()


def load_checkpoint(torch, path, device):
    """Checkpoint path and device -> safely loaded training checkpoint."""
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location=device, weights_only=True)


def build_model(torch, device, state):
    """Official checkpoint and device -> strictly loaded inference model."""
    from model import D2Mamba

    model = D2Mamba(base_dim=24, num_paths=4, mamba_layers=4, scan_strategy_name="astar").to(device)
    state = dict(state.get("model_state_dict", state))
    for name in ("pathfinder", "saliency_head", "mamba_corrector"):
        if getattr(model.scan_strategy, name) is not getattr(model, name):
            raise RuntimeError("D2Mamba scan strategy must share its original model parameters.")
    for key in model.state_dict():
        if key.startswith("scan_strategy.") and key not in state:
            original = key.removeprefix("scan_strategy.")
            if original in state:
                state[key] = state[original]
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensor in [0, 1] -> enhanced RGB tensor in [0, 1]."""
    return model(tensor)[0].clamp(0, 1)
