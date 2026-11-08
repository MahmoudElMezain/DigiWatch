"""Run the official PSPL HRNet checkpoint with its original normalized RGB input and haze-removal branch."""
CHECKPOINT_FILENAME = "PSPL_HRNet_net_G.pth"
DEPENDENCIES = ()


def build_model(torch, device, state):
    """Torch, device, and state dictionary -> loaded PSPL HRNet."""
    from models.semi.HFBranch import HFBranch

    model = HFBranch().to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> enhanced RGB tensors in [0, 1]."""
    _, enhanced = model(tensor * 2 - 1)
    return ((enhanced + 1) / 2).clamp(0, 1)
