"""ImageDehazer enhances RGB images or patches with a supported pretrained underwater model, loads its weights once, and returns 256 by 256 RGB images."""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import threading
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


UIERL_CONTEXT_DIR = None  # Folder containing other related photographs.
UIERL_CONTEXT_COUNT = 2  # Additional photographs per UIERL input.
UIERL_MAX_CANDIDATES = 128  # Maximum photographs considered for selection.

_IMPORT_LOCK = threading.RLock()
_DEPTH_REPOSITORY = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _uierl_descriptor(rgb):
    """RGB image -> thumbnail fingerprint and normalized appearance features."""
    thumbnail = np.asarray(rgb.resize((64, 64), Image.Resampling.BILINEAR))
    histogram = np.concatenate([np.histogram(thumbnail[:, :, channel], 16, (0, 256))[0]
                                for channel in range(3)]).astype(np.float32)
    histogram /= histogram.sum()
    spatial = np.asarray(rgb.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return hashlib.sha256(thumbnail.tobytes()).digest(), np.sqrt(histogram), spatial


def _uierl_regions(depth, order):
    """Finite depth array and label order -> aligned three-cluster uint8 labels."""
    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("UIERL requires a finite two-dimensional depth prediction.")
    unique = np.unique(values)
    if unique.size < 3:
        raise ValueError("The depth estimator produced fewer than three distinct depths.")
    if order not in {"ascending", "descending"}:
        raise ValueError("depth_label_order must be 'ascending' or 'descending'.")
    flat = ((values - unique[0]) / (unique[-1] - unique[0])).reshape(-1)
    unique = (unique - unique[0]) / (unique[-1] - unique[0])
    centers = unique[np.linspace(0, unique.size - 1, 3, dtype=int)]
    for _ in range(100):
        labels = np.abs(flat[:, None] - centers).argmin(axis=1)
        updated = np.array([flat[labels == index].mean() for index in range(3)])
        if not np.isfinite(updated).all():
            raise ValueError("UIERL depth clustering produced an empty region.")
        if np.max(np.abs(updated - centers)) < 1e-7:
            centers = updated
            break
        centers = updated
    labels = np.abs(flat[:, None] - centers).argmin(axis=1)
    palette = np.array([0, 128, 255], dtype=np.uint8)
    return (palette if order == "ascending" else palette[::-1])[labels].reshape(values.shape)


class _UIERLInputs:
    def __init__(self, torch, device, weights_directory, options):
        """Runtime and input settings -> cached automatic UIERL preprocessing."""
        self.torch, self.device, self.options = torch, device, options
        self.weights_directory = weights_directory
        self.depth_model = self.processor = None
        self.candidate_key, self.candidates = None, []
        self.images, self.regions = {}, OrderedDict()
        self.metadata = {"depth_model": _DEPTH_REPOSITORY, "depth_revision": _DEPTH_REVISION,
                         "grouping": "color_histogram_and_spatial_similarity",
                         "regions": "three_cluster_kmeans", "original_preprocessing": False}

    @staticmethod
    def _positive(value, name):
        """Value and parameter name -> validated positive integer."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    def _candidates(self):
        """Configured folder or image list -> cached candidate appearance records."""
        supplied = self.options.get("context_images")
        folder = self.options.get("context_dir")
        if isinstance(supplied, (str, Path)):
            supplied = [supplied]
        if supplied:
            paths = [Path(path).expanduser().resolve() for path in supplied]
        else:
            if not folder:
                raise ValueError("Set UIERL_CONTEXT_DIR in Image Dehazer.py to a folder of other related photographs.")
            folder = Path(folder).expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError(f"UIERL context folder not found: {folder}")
            paths = sorted((path for path in folder.iterdir() if path.is_file()
                            and path.suffix.lower() in _IMAGE_SUFFIXES), key=lambda path: str(path).casefold())
        limit = self._positive(self.options.get("max_context_candidates", UIERL_MAX_CANDIDATES), "max_context_candidates")
        if len(paths) > limit:
            paths = [paths[index] for index in np.linspace(0, len(paths) - 1, limit, dtype=int)]
        source = self.options.get("source_image")
        source = Path(source).expanduser().resolve() if source else None
        key = (tuple(str(path) for path in paths), str(source))
        if key == self.candidate_key:
            return self.candidates
        records = []
        for path in paths:
            if path == source:
                continue
            try:
                with Image.open(path) as image:
                    if image.mode in {"1", "L", "I", "F"}:
                        continue
                    image.draft("RGB", (128, 128))
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    descriptor = _uierl_descriptor(rgb)
                records.append((path, descriptor))
            except (OSError, ValueError):
                continue
        self.candidate_key, self.candidates = key, records
        return records

    def _image(self, path):
        """Context path -> cached 256-pixel RGB image and pixel digest."""
        status = path.stat()
        key = (str(path), status.st_size, status.st_mtime_ns)
        if key not in self.images:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
                array = np.array(rgb, dtype=np.uint8)
            self.images[key] = (array, hashlib.sha256(array.tobytes()).hexdigest())
        return self.images[key]

    def _select(self, target):
        """Prepared target pixels -> distinct related RGB images and identities."""
        fingerprint, histogram, spatial = _uierl_descriptor(Image.fromarray(target))
        ranked = []
        for path, (candidate_fingerprint, candidate_histogram, candidate_spatial) in self._candidates():
            if candidate_fingerprint == fingerprint:
                continue
            score = 0.7 * np.linalg.norm(candidate_histogram - histogram) / np.sqrt(2)
            score += 0.3 * np.abs(candidate_spatial - spatial).mean()
            ranked.append((float(score), str(path).casefold(), path))
        count = self._positive(self.options.get("context_count", UIERL_CONTEXT_COUNT), "context_count")
        selected, identities = [], []
        seen = {hashlib.sha256(target.tobytes()).hexdigest()}
        for _, _, path in sorted(ranked):
            image, digest = self._image(path)
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(image)
            identities.append({"filename": path.name, "prepared_pixels_sha256": digest})
            if len(selected) == count:
                break
        if not selected:
            raise ValueError("UIERL needs at least one different related photograph in UIERL_CONTEXT_DIR.")
        self.metadata["context_images"] = identities
        self.metadata["context_count"] = len(selected)
        self.metadata["selection_reference"] = "prepared_target_pixels"
        return selected

    def _load_depth(self):
        """Load the official pretrained depth estimator once, without incomplete state loading."""
        if self.depth_model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("UIERL needs transformers and safetensors. Install: python -m pip install transformers==4.44.2 safetensors") from exc
        directory = Path(self.options.get("depth_model_path") or self.weights_directory / "depth_anything_v2_hf")
        required = ("config.json", "preprocessor_config.json", "model.safetensors")
        if not all((directory / filename).is_file() for filename in required):
            if self.options.get("depth_model_path"):
                raise FileNotFoundError(f"The local UIERL depth model is incomplete: {directory}")
            snapshot_download(repo_id=_DEPTH_REPOSITORY, revision=_DEPTH_REVISION,
                              local_dir=directory, allow_patterns=list(required))
        self.processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True)
        model, information = AutoModelForDepthEstimation.from_pretrained(
            directory, local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        if any(information.values()):
            raise ValueError(f"The UIERL depth estimator did not load strictly: {information}")
        self.depth_model = model.to(self.device).eval()
        with (directory / "model.safetensors").open("rb") as stream:
            self.metadata["depth_checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()

    def _region_map(self, image):
        """Prepared RGB pixels -> cached, aligned learned-depth region labels."""
        order = self.options.get("depth_label_order", "ascending")
        key = (hashlib.sha256(image.tobytes()).digest(), order)
        if key not in self.regions:
            inputs = self.processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
            depth = self.depth_model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                depth[:, None], size=(256, 256), mode="bicubic", align_corners=False,
            )[0, 0].detach().cpu().numpy()
            self.regions[key] = _uierl_regions(depth, order)
            while len(self.regions) > 64:
                self.regions.popitem(last=False)
        self.regions.move_to_end(key)
        return self.regions[key]

    def prepare_group(self, target):
        """Prepared target tensor -> related RGB group and aligned three-region masks."""
        pixels = target[0].detach().cpu().permute(1, 2, 0).numpy()
        pixels = np.rint(pixels * 255).clip(0, 255).astype(np.uint8)
        images = [pixels, *self._select(pixels)]
        self._load_depth()
        maps = [self._region_map(image) for image in images]
        contexts = [self.torch.from_numpy(image.copy()).permute(2, 0, 1).to(self.device).float() / 255
                    for image in images[1:]]
        group = self.torch.cat([target, *(image.unsqueeze(0) for image in contexts)], dim=0)
        regions = self.torch.from_numpy(np.stack(maps)).unsqueeze(1).to(self.device)
        self.metadata["depth_label_order"] = self.options.get("depth_label_order", "ascending")
        self.metadata["target_prepared_pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        return group, regions


class _RepositoryImports:
    def __init__(self, root: Path):
        """Store a repository path; returns None."""
        self.root = root
        self.modules = {}

    @contextmanager
    def activate(self):
        """Yield imports from the stored repository."""
        with _IMPORT_LOCK:
            names = {
                p.stem if p.is_file() else p.name
                for p in self.root.iterdir()
                if (p.is_file() and p.suffix == ".py") or p.is_dir()
            }

            def belongs(name):
                """Take a module name; return whether it belongs to this repository."""
                return name.split(".", 1)[0] in names

            previous_modules = {k: v for k, v in list(sys.modules.items()) if belongs(k)}
            previous_path = sys.path[:]
            for name in previous_modules:
                del sys.modules[name]
            sys.modules.update(self.modules)
            sys.path.insert(0, str(self.root))
            try:
                yield
            finally:
                self.modules = {k: v for k, v in list(sys.modules.items()) if belongs(k)}
                for name in self.modules:
                    del sys.modules[name]
                sys.modules.update(previous_modules)
                sys.path[:] = previous_path


class ImageDehazer:
    @classmethod
    def model_registry(cls):
        """Return model definitions and verified availability."""
        path = Path(__file__).resolve().parent / "Models" / "registry.json"
        return {entry["key"]: entry for entry in json.loads(path.read_text(encoding="utf-8"))["models"]}

    @classmethod
    def supported_models(cls):
        """Return the names of the models available in this pipeline."""
        return tuple(entry["name"] for entry in cls.model_registry().values() if entry["available"])

    def __init__(
        self,
        model_name: str = "mulagan",  # Name from supported_models().
        device: str = "auto",  # auto, cpu, or cuda.
        repo_root=None,  # Optional architecture folder.
        weights_path=None,  # Optional checkpoint path.
        image_size: int = 256,
        diffusion_steps: int | None = None,  # None uses each model's default.
        seed: int | None = 1234,  # Seed for stochastic models.
        deterministic: bool = False,  # Enable repeatable inference kernels.
        model_options: dict | None = None,  # Optional model-specific inputs.
    ):
        """Configure the model and inference parameters; returns None."""
        canonical = "".join(c for c in str(model_name).lower() if c.isalnum())
        registry = self.model_registry()
        definition = registry.get(canonical)
        if definition is None or not definition["available"]:
            reason = f"{definition['reason']} " if definition and definition.get("reason") else ""
            raise ValueError(f"'{model_name}' is not available in the current pipeline. {reason}"
                             f"Available models: {', '.join(self.supported_models())}.")
        if image_size != 256:
            raise ValueError("The supplied pretrained model adapters require image_size=256.")
        if diffusion_steps is None:
            diffusion_steps = definition.get("diffusion_steps", 2000)
        if isinstance(diffusion_steps, bool) or not isinstance(diffusion_steps, int) or diffusion_steps < 1:
            raise ValueError("diffusion_steps must be a positive integer.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32):
            raise ValueError("seed must be None or an integer in [0, 2**32).")
        self.model_name = canonical
        self.device_mode = str(device).strip().lower()
        if self.device_mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
        self.definition = definition
        default_root = Path(__file__).resolve().parent
        self.model_root = default_root / "Models" / definition["directory"]
        self.repo_root = (Path(repo_root) if repo_root is not None else self.model_root / "architecture").resolve()
        self.weights_path = (Path(weights_path) if weights_path is not None else self.model_root / "weights" / definition["checkpoint"]).resolve()
        self.weights_directory = self.weights_path.parents[len(Path(definition["checkpoint"]).parts) - 1]
        self._adapter = None
        self.image_size = image_size
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        if not isinstance(deterministic, bool):
            raise ValueError("deterministic must be a boolean.")
        self.deterministic = deterministic
        if model_options is not None and not isinstance(model_options, dict):
            raise ValueError("model_options must be a dictionary or None.")
        self.model_options = copy.deepcopy(model_options or {})
        if canonical == "uierl":
            folder = os.environ.get("UIERL_CONTEXT_DIR") or UIERL_CONTEXT_DIR
            self.model_options.setdefault("context_dir", str(folder) if folder else None)
            self.model_options.setdefault("context_count", UIERL_CONTEXT_COUNT)
            if not self.model_options["context_dir"] and self.model_options.get("source_image"):
                self.model_options["context_dir"] = str(Path(self.model_options["source_image"]).resolve().parent)
            if not self.model_options["context_dir"] and not self.model_options.get("context_images"):
                raise ValueError("Set UIERL_CONTEXT_DIR in Image Dehazer.py to a folder containing other related photographs.")
        self._uierl_inputs = None
        self.model = None
        self.device = None
        self._imports = _RepositoryImports(self.repo_root)
        self._torch = None
        self._transforms = None
        self._config = None
        self._cublas_workspace_config = None

    @property
    def loaded(self):
        """Return whether the model has been loaded."""
        return self.model is not None

    def required_weight_paths(self):
        """Return the required checkpoint paths, including additional model stages."""
        files = self.definition.get("checkpoints", {"model": self.definition["checkpoint"]})
        return {name: self.weights_path if filename == self.definition["checkpoint"]
                else self.weights_directory / filename for name, filename in files.items()}

    def _prepare_deterministic_environment(self):
        """Prepare deterministic CUDA settings; returns None."""
        if self.deterministic and self.device_mode != "cpu":
            self._cublas_workspace_config = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def load(self):
        """Load the selected checkpoint and return this ImageDehazer instance."""
        if self.loaded:
            return self
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"{self.model_name} repository not found: {self.repo_root}")
        missing = [str(path) for path in self.required_weight_paths().values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.definition['name']} model files not found: {', '.join(missing)}")
        self._prepare_deterministic_environment()
        try:
            import torch
            from torchvision import transforms
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                "Cannot import compatible PyTorch/torchvision. Run with the selected "
                f"{self.model_name} conda environment. Original error: {exc}"
            ) from exc
        self._torch, self._transforms = torch, transforms
        if self.device_mode == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is unavailable in this environment.")
        self.device = torch.device(
            "cuda" if self.device_mode == "auto" and torch.cuda.is_available()
            else "cpu" if self.device_mode == "auto" else self.device_mode
        )

        with self._imports.activate():
            if self.definition["adapter"] != "builtin":
                path = self.model_root / self.definition["adapter"]
                spec = importlib.util.spec_from_file_location(f"_coral_adapter_{self.model_name}", path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load model adapter: {path}")
                self._adapter = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = self._adapter
                spec.loader.exec_module(self._adapter)
            if self._adapter is not None and hasattr(self._adapter, "load_checkpoint"):
                state = self._adapter.load_checkpoint(torch, self.weights_path, self.device)
            elif self._adapter is not None and self.definition.get("checkpoints"):
                state = {name: torch.load(path, map_location=self.device, weights_only=True)
                         for name, path in self.required_weight_paths().items()}
            else:
                state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            if self._adapter is None:
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not isinstance(state, dict) or not state:
                    raise ValueError(f"Expected a non-empty model state dictionary: {self.weights_path}")
                if all(key.startswith("module.") for key in state):
                    state = {key[len("module."):]: value for key, value in state.items()}

            if self._adapter is not None:
                model = self._adapter.build_model(torch, self.device, state)
            elif self.model_name == "uierl":
                module = importlib.import_module("models.networks")

                def projection(C_Channel, Resized_C):
                    """Group features and channel count -> original device-portable projection."""
                    convolution = torch.nn.Conv2d(C_Channel.shape[1], Resized_C, 1, 1, 0)
                    return convolution.to(device=C_Channel.device, dtype=C_Channel.dtype)(C_Channel)

                module.Conv_Channel = projection
                model = module.define_G(3, 3, "unet").to(self.device)
            elif self.model_name == "mulagan":
                module = importlib.import_module("models.MuLA_GAN")
                model = module.MuLA_GAN_Generator().to(self.device)
            elif self.model_name == "semiuir":
                module = importlib.import_module("model")
                model = module.AIMnet().to(self.device)
            elif self.model_name == "dgdcgan":
                module = importlib.import_module("nets.networks_Dgd")
                model = module.ResUnet1().to(self.device)
            elif self.model_name == "clipuie":
                config_path = self.repo_root / "config" / "sr_sr3_32_256_CLIP-UIE.json"
                lines = config_path.read_text(encoding="utf-8").splitlines()
                self._config = json.loads("\n".join(line.split("//", 1)[0] for line in lines))
                options = self._config["model"]
                if int(options["diffusion"]["image_size"]) != self.image_size:
                    raise ValueError("CLIP-UIE config image_size must match image_size=256.")
                unet = importlib.import_module("model.sr3_modules.unet")
                diffusion = importlib.import_module("model.sr3_modules.diffusion")
                settings = options["unet"]
                denoiser = unet.UNet(
                    in_channel=settings["in_channel"], out_channel=settings["out_channel"],
                    norm_groups=settings.get("norm_groups") or 32, inner_channel=settings["inner_channel"],
                    channel_mults=settings["channel_multiplier"], attn_res=settings["attn_res"],
                    res_blocks=settings["res_blocks"], dropout=settings["dropout"], image_size=self.image_size)
                model = diffusion.GaussianDiffusion(
                    denoiser, image_size=self.image_size, channels=options["diffusion"]["channels"],
                    conditional=options["diffusion"]["conditional"], inference_only=True).to(self.device)
                state = {key: value for key, value in state.items() if key.startswith("denoise_fn.")}
                if not state:
                    raise ValueError("CLIP-UIE checkpoint has no denoise_fn weights.")
            elif self.model_name == "dcduie":
                config_path = self.repo_root / "config" / "UIE.json"
                with config_path.open(encoding="utf-8") as stream:
                    self._config = json.load(stream)
                self._config.update(phase="val", distributed=False,
                                    gpu_ids=[0] if self.device.type == "cuda" else None)
                if int(self._config["model"]["diffusion"]["image_size"]) != self.image_size:
                    raise ValueError("DCD-UIE config image_size must match image_size=256.")
                module = importlib.import_module("model.networks")
                model = module.define_G(self._config).to(self.device)

                schedule_buffers = {
                    "betas", "alphas_cumprod", "alphas_cumprod_prev",
                    "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
                    "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
                    "sqrt_recipm1_alphas_cumprod", "posterior_variance",
                    "posterior_log_variance_clipped", "posterior_mean_coef1",
                    "posterior_mean_coef2",
                }
                state = {key: value for key, value in state.items() if key not in schedule_buffers}

            if self._adapter is None:
                model.load_state_dict(state, strict=True)
            model.eval()
            if self.model_name in {"dcduie", "clipuie"}:
                schedule = copy.deepcopy(self._config["model"]["beta_schedule"]["val"])
                schedule["n_timestep"] = self.diffusion_steps
                model.set_new_noise_schedule(schedule, device=self.device)
        self.model = model
        return self

    @staticmethod
    def _rgb(image):
        """Take a PIL image or uint8 RGB array; return an RGB PIL image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected a PIL image or uint8 RGB array with shape (height, width, 3).")
        if not array.shape[0] or not array.shape[1]:
            raise ValueError("Image must have nonzero width and height.")
        return Image.fromarray(array)

    @staticmethod
    def luminance_estimation(image):
        """Take an RGB image; return its Semi-UIR illumination image."""
        rgb = image.convert("RGB")
        illuminance = np.ones_like(np.asarray(rgb, dtype=np.float32))
        for sigma in (15, 60, 90):
            blurred = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32)
            illuminance += np.clip(np.log10(blurred + 1e-8), 0, 255)
        illuminance /= 3.0
        normalized = ((illuminance - np.min(illuminance)) /
                      (np.max(illuminance) - np.min(illuminance) + 1e-6))
        return Image.fromarray(np.uint8(normalized * 255.0))

    @contextmanager
    def _deterministic_state(self):
        """Yield deterministic inference settings and restore previous flags."""
        if not self.deterministic:
            yield
            return
        torch = self._torch
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        benchmark = torch.backends.cudnn.benchmark
        cudnn_deterministic = torch.backends.cudnn.deterministic
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            yield
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic

    @contextmanager
    def _random_state(self):
        """Yield seeded inference settings and restore previous random states."""
        if self.seed is None or self.model_name == "mulagan":
            yield
            return
        torch = self._torch
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(self.seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def enhance(self, image):
        """Take a PIL image or uint8 RGB array; return a 256 by 256 RGB PIL image."""
        rgb = self._rgb(image)
        self.load()
        torch, transforms = self._torch, self._transforms
        bicubic = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
        rgb = transforms.Resize((self.image_size, self.image_size), bicubic)(rgb)
        tensor = transforms.ToTensor()(rgb)
        with self._imports.activate(), self._deterministic_state(), self._random_state(), torch.no_grad():
            if self.model_name == "uierl":
                batch = tensor.unsqueeze(0).to(self.device)
                devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    if self._uierl_inputs is None:
                        self._uierl_inputs = _UIERLInputs(torch, self.device, self.weights_directory, self.model_options)
                    group, regions = self._uierl_inputs.prepare_group(batch)
                normalized = group * 2 - 1
                masks = [[(regions[index:index + 1] == label).to(dtype=batch.dtype)
                          for label in (0, 128, 255)] for index in range(group.shape[0])]
                masked = [[normalized[index:index + 1] * mask for mask in image_masks]
                          for index, image_masks in enumerate(masks)]
                prediction = self.model(normalized, masks, masked)[0].detach().cpu()
                if not torch.isfinite(prediction).all():
                    raise ValueError("UIERL returned non-finite image values.")
                prediction = ((prediction + 1) / 2).clamp(0, 1)
            elif self._adapter is not None:
                prediction = self._adapter.predict(self.model, tensor.unsqueeze(0).to(self.device), torch, {
                    "image_size": self.image_size, "diffusion_steps": self.diffusion_steps,
                    "seed": self.seed, "device": self.device, "weights_path": str(self.weights_path),
                    "model_options": self.model_options,
                })
                if not torch.is_tensor(prediction) or prediction.shape != (1, 3, self.image_size, self.image_size):
                    raise ValueError(f"{self.definition['name']} must return a 1 x 3 x 256 x 256 RGB tensor.")
                if not torch.isfinite(prediction).all():
                    raise ValueError(f"{self.definition['name']} returned non-finite image values.")
                prediction = prediction[0].detach().cpu().clamp(0, 1)
            elif self.model_name == "mulagan":
                tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(tensor)
                prediction = self.model(tensor.unsqueeze(0).to(self.device)).squeeze(0).cpu()
                prediction = (prediction * 0.5 + 0.5).clamp(0, 1)
            elif self.model_name == "semiuir":
                illumination = transforms.ToTensor()(self.luminance_estimation(rgb))
                prediction, _ = self.model(tensor.unsqueeze(0).to(self.device),
                                           illumination.unsqueeze(0).to(self.device))
                prediction = prediction[0].detach().cpu().clamp(0, 1)
            elif self.model_name == "dgdcgan":
                prediction = self.model(tensor.unsqueeze(0).to(self.device))[0].detach().cpu().clamp(0, 1)
            elif self.model_name in {"dcduie", "clipuie"}:
                prediction = self.model.super_resolution(
                    (tensor * 2.0 - 1.0).unsqueeze(0).to(self.device), continous=False)
                prediction = prediction.detach().cpu().clamp(-1, 1)
                if prediction.dim() == 4:
                    prediction = prediction[0]
                prediction = (prediction + 1.0) / 2.0
        if self.definition.get("output_rounding") == "round":
            pixels = prediction.permute(1, 2, 0).float().numpy()
            return Image.fromarray(np.rint(pixels * 255.0).astype(np.uint8))
        return transforms.ToPILImage()(prediction)

    def enhance_many(self, images: Iterable):
        """Take an iterable of RGB images; return enhanced PIL images in order."""
        return [self.enhance(image) for image in images]

    dehaze = enhance

    def to_dict(self):
        """Return model paths and inference settings as a dictionary."""
        return {
            "model": self.model_name,
            "repo_root": str(self.repo_root),
            "weights_path": str(self.weights_path),
            "image_size": self.image_size,
            "device": str(self.device) if self.device is not None else self.device_mode,
            "diffusion_steps": self.diffusion_steps if "diffusion_steps" in self.definition else None,
            "seed": self.seed if self.model_name != "mulagan" else None,
            "deterministic": self.deterministic,
            "model_options": copy.deepcopy(self.model_options),
            "uierl_preprocessing": copy.deepcopy(self._uierl_inputs.metadata) if self._uierl_inputs else None,
            "cublas_workspace_config": self._cublas_workspace_config,
        }
