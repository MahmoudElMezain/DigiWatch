"""Load the official WaterMamba model and enhance RGB tensors using its selective-scan architecture."""

CHECKPOINT_FILENAME = "net_g_232000.pth"
DEPENDENCIES = ("einops", "timm")


def build_model(torch, device, state):
    """Official checkpoint and device -> strictly loaded WaterMamba model."""
    from watermamba import WaterMamba

    model = WaterMamba(inp_channels=3, out_channels=3, dim=24, num_blocks=[1, 1, 1, 1],
                       mam_blocks=[2, 2, 2, 2], d_state=16, drop_path=0,
                       attn_drop_rate=0, resolution=[256, 256], expand=2, d_conv=4, bias=False).to(device)
    model.load_state_dict(state.get("params", state), strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensor in [0, 1] -> enhanced RGB tensor in [0, 1]."""
    return model(tensor).clamp(0, 1)
