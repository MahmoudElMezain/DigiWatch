"""Load SLURPP's dual diffusion networks and cross-latent decoder with local Stable Diffusion 2 components, then return the restored clear image."""
from pathlib import Path
import json

CHECKPOINT_FILENAME = "slurpp/config.yaml"
CHECKPOINT_FILENAMES = (
    CHECKPOINT_FILENAME,
    "slurpp/checkpoint/diffusion/unet1/config.json",
    "slurpp/checkpoint/diffusion/unet1/diffusion_pytorch_model.bin",
    "slurpp/checkpoint/diffusion/unet2/config.json",
    "slurpp/checkpoint/diffusion/unet2/diffusion_pytorch_model.bin",
    "slurpp/checkpoint/cld/cld_clear.pth",
    "stable-diffusion-2/scheduler/scheduler_config.json",
    "stable-diffusion-2/text_encoder/config.json",
    "stable-diffusion-2/text_encoder/model.safetensors",
    "stable-diffusion-2/tokenizer/merges.txt",
    "stable-diffusion-2/tokenizer/special_tokens_map.json",
    "stable-diffusion-2/tokenizer/tokenizer_config.json",
    "stable-diffusion-2/tokenizer/vocab.json",
    "stable-diffusion-2/vae/config.json",
    "stable-diffusion-2/vae/diffusion_pytorch_model.safetensors",
)
DEPENDENCIES = ("diffusers", "transformers", "accelerate", "safetensors", "omegaconf")
DEFAULT_DIFFUSION_STEPS = 1


def load_checkpoint(torch, path, device):
    """Local release configuration path -> validated multi-file asset location."""
    root = Path(path).resolve().parent.parent
    missing = [filename for filename in CHECKPOINT_FILENAMES if not (root / filename).is_file()]
    if missing:
        raise FileNotFoundError("SLURPP model files are missing: " + ", ".join(missing))
    return {"weights_root": root}


def build_model(torch, device, state):
    """Local release assets and device -> strictly loaded SLURPP wrapper."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from safetensors.torch import load_file
    from omegaconf import OmegaConf
    import numpy as np
    from slurpp_runtime.dual_unet_condition import DualUNetCondition
    from slurpp_runtime.CLUNet import CrossLatentUNet
    from slurpp_runtime.slurpp_pipeline import SlurppPipeline

    root = state["weights_root"]
    release = root / "slurpp"
    base = root / "stable-diffusion-2"
    config = OmegaConf.load(release / "config.yaml")
    if (not config.get("dual") or list(config.trainer.inputs) != ["u"]
            or list(config.trainer.inputs2) != ["u"] or list(config.trainer.output) != ["clear"]
            or list(config.trainer.output2) != ["bc", "ill"] or not config.get("one_step")):
        raise ValueError("SLURPP requires the official dual-network release configuration.")
    networks = []
    for number in (1, 2):
        directory = release / "checkpoint" / "diffusion" / f"unet{number}"
        settings = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        networks.append(UNet2DConditionModel.from_config(settings))
    dual = DualUNetCondition(unet1=networks[0], unet2=networks[1])
    dual.add_additional_params()
    for number, network in enumerate(networks, 1):
        checkpoint = release / "checkpoint" / "diffusion" / f"unet{number}" / "diffusion_pytorch_model.bin"
        tensors = torch.load(checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(tensors, strict=True)
        network.to(device)
        del tensors
    vae = AutoencoderKL.from_config(json.loads((base / "vae/config.json").read_text(encoding="utf-8")))
    vae_state = load_file(str(base / "vae/diffusion_pytorch_model.safetensors"))
    vae._convert_deprecated_attention_blocks(vae_state)
    vae.load_state_dict(vae_state, strict=True)
    text_encoder = CLIPTextModel(CLIPTextConfig.from_json_file(str(base / "text_encoder/config.json")))
    text_state = load_file(str(base / "text_encoder/model.safetensors"))
    position_ids = text_state.pop("text_model.embeddings.position_ids", None)
    if position_ids is not None and not torch.equal(position_ids, text_encoder.text_model.embeddings.position_ids):
        raise ValueError("SLURPP text-encoder position indices do not match its configuration.")
    text_encoder.load_state_dict(text_state, strict=True)
    tokenizer = CLIPTokenizer.from_pretrained(str(base / "tokenizer"), local_files_only=True)
    scheduler = DDIMScheduler.from_config(json.loads((base / "scheduler/scheduler_config.json").read_text(encoding="utf-8")))
    scheduler.register_to_config(timestep_spacing="trailing")
    cld = CrossLatentUNet(config_path=str(base / "vae/config.json"))
    numeric_types = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                     type(np.dtype("float64")), type(np.dtype("float32"))]
    with torch.serialization.safe_globals(numeric_types):
        cld_state = torch.load(release / "checkpoint/cld/cld_clear.pth", map_location="cpu", weights_only=True)
    cld.load_state_dict(cld_state["state_dict"], strict=True)
    pipeline = SlurppPipeline(dual, vae, scheduler, text_encoder, tokenizer)
    pipeline.skip_connection = True
    pipeline.vae_cld = cld.to(device)
    pipeline = pipeline.to(device)

    class SLURPPModel(torch.nn.Module):
        def __init__(self):
            """Register inference modules and their shared diffusion pipeline."""
            super().__init__()
            self.unet = pipeline.unet
            self.vae = pipeline.vae
            self.text_encoder = pipeline.text_encoder
            self.cld = pipeline.vae_cld
            self.pipeline = pipeline

    return SLURPPModel().eval()


def predict(model, tensor, torch, options):
    """One RGB tensor in [0, 1] -> restored clear RGB tensor in [0, 1]."""
    if tensor.shape[0] != 1:
        raise ValueError("The official SLURPP inference path accepts one image at a time.")
    normalized = tensor * 2 - 1
    output = model.pipeline([normalized, normalized],
                            denoising_steps=options.get("diffusion_steps", DEFAULT_DIFFUSION_STEPS),
                            show_progress_bar=False, is_dual=True)
    return output[:1].clamp(0, 1)
