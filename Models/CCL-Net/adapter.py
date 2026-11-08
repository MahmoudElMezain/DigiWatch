"""Run the official CCL-Net color-correction and haze-removal stages using their original Lab preprocessing and both trained generators."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

CHECKPOINT_FILENAME = "best_net_G.pth"
CHECKPOINT_FILENAMES = {"hr": "best_net_G.pth", "cc": "best_net_G_CC.pth"}
DEPENDENCIES = ("opencv-python", "numpy")


def build_model(torch, device, state):
    """Torch, device, and two state dictionaries -> loaded CCL-Net stages."""
    from models.CCLNet.CCNet import CCNet
    from models.CCLNet.HRNet import HRNet
    from models.CCLNet.Public.util.LAB2RGB_v2 import Lab2RGB

    options = SimpleNamespace(init_type="normal", init_gain=0.02, gpu_ids=[])
    with redirect_stdout(StringIO()):
        model = torch.nn.ModuleDict({"cc": CCNet(options), "hr": HRNet(options)}).to(device)
    for name in ("cc", "hr"):
        model[name].load_state_dict(state[name], strict=True)
    model.lab2rgb = Lab2RGB(useGPU=False)
    for name, value in vars(model.lab2rgb).items():
        if torch.is_tensor(value):
            setattr(model.lab2rgb, name, value.to(device))
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> enhanced RGB tensors in [0, 1]."""
    import cv2
    import numpy as np

    images = (tensor.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    normalized = []
    for image in images:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab).astype(np.float32)
        for channel, (minimum, scale) in enumerate(zip((0., 42., 20.), (255., 184., 203.))):
            lab[:, :, channel] = (lab[:, :, channel] - minimum) / scale
        normalized.append(np.uint8(lab * 255))
    lab_tensor = torch.from_numpy(np.stack(normalized)).permute(0, 3, 1, 2).to(tensor.device).float()
    lab_tensor = lab_tensor / 255 * 2 - 1
    _, corrected_lab, _ = model["cc"](lab_tensor)
    corrected_rgb = model.lab2rgb.labn12p1_to_rgbn12p1(corrected_lab)
    enhanced, _, _ = model["hr"](corrected_rgb)
    return ((enhanced + 1) / 2).clamp(0, 1)
