"""Run the official UVE-Net large model on a single-frame sequence without inventing or repeating neighboring video frames."""
CHECKPOINT_FILENAME = "large_net_g.pth"
DEPENDENCIES = ("einops",)


def build_model(torch, device, state):
    """Torch, device, and state dictionary -> loaded large UVE-Net."""
    from uve_net import Deblur

    state = state.get("params", state)
    model = Deblur(num_feat=state["feat_extractor.weight"].shape[0]).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensors in [0, 1] -> enhanced single-frame RGB tensors."""
    enhanced, _ = model(tensor.unsqueeze(1))
    return enhanced[:, 0].clamp(0, 1)
