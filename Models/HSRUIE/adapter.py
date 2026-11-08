"""Load the five official HSRUIE inference components and restore normalized RGB images without initializing training losses or CLIP."""
from __future__ import annotations

CHECKPOINT_FILENAME = "latest_net_NC.pth"
CHECKPOINT_FILENAMES = {
    "nc": "latest_net_NC.pth",
    "high2": "latest_net_HIGH2.pth",
    "high3": "latest_net_HIGH3.pth",
    "sad": "latest_net_SAD.pth",
    "vgg": "vgg19-dcbb9e9d.pth",
}
DEPENDENCIES = ()


def load_checkpoint(torch, path, device):
    """Primary checkpoint path -> safely loaded dictionaries for all inference components."""
    states = {}
    for name, filename in CHECKPOINT_FILENAMES.items():
        file = path if name == "nc" else path.parent / filename
        if not file.is_file():
            raise FileNotFoundError(f"HSRUIE requires {file}")
        states[name] = torch.load(file, map_location="cpu", weights_only=True)
    return states


def build_model(torch, device, state):
    """Five component state dictionaries -> strictly loaded HSRUIE model."""
    from torchvision.models import vgg19
    from hsruie import HSRUIE

    vgg = vgg19(weights=None)
    vgg.load_state_dict(state["vgg"], strict=True)
    model = HSRUIE(vgg.features).to(device)
    for name in ("nc", "high2", "high3", "sad"):
        getattr(model, name).load_state_dict(state[name], strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> enhanced RGB tensors in [0, 1]."""
    return ((model(tensor * 2.0 - 1.0) + 1.0) / 2.0).clamp(0, 1)
