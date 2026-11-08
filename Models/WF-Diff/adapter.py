"""Run the official WF-Diff wavelet predictor and residual diffusion reconstruction on RGB tensors."""

CHECKPOINT_FILENAME = "net_g_405000.pth"
DEPENDENCIES = ("einops",)
DEFAULT_DIFFUSION_STEPS = 10


def build_model(torch, device, state):
    """Official state dictionary and device -> strictly loaded WF-Diff model."""
    from basicsr.archs.wfdiffx2_arch import WfDiffx2

    schedule = {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2}
    model = WfDiffx2(inner_channel=48, norm_groups=24, schedule_opt=schedule, sample_proc="ddim").to(device)
    model.load_state_dict(state.get("params", state), strict=True)
    return model.eval()


def predict(model, tensor, torch, options):
    """RGB tensor in [0, 1] -> reconstructed RGB tensor in [0, 1]."""
    from basicsr.archs.Padiff_arch.wavelet import DWT, IWT

    if options.get("diffusion_steps", DEFAULT_DIFFUSION_STEPS) != DEFAULT_DIFFUSION_STEPS:
        raise ValueError("WF-Diff uses the official fixed 10-step DDIM sampler; select diffusion_steps=10.")
    preliminary, residual_low, _, residual_high, _, _, _ = model(tensor, None)
    bands = DWT()(preliminary)
    batch = preliminary.shape[0]
    reconstructed = IWT()(torch.cat((bands[:batch] + residual_low, bands[batch:] + residual_high), dim=0))
    return reconstructed.clamp(0, 1)
