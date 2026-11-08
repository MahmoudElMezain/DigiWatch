"""Analyze manually selected or automatically ranked CoralSCOP predictions with an underwater enhancement model and save one figure per coral showing the input image, selected coral, and CoralWatch bleaching estimate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

INPUT_IMAGE = None  # Set an input image name or path.
PREPARATION_STRATEGY = "coral-resize"  # full-resize, coral-resize, patches, overlap.
PATCH_SIZE = 256  # Source patch size; model inputs are resized to 256.
PATCH_GRID = None #(2, 2), (4, 3), (8, 6), or None for native size.
PATCH_STRIDE = 128  # Overlap stride in source pixels.
DEHAZING_MODEL = "clip-uie"  # Model name from Models/registry.json.
ANALYSIS_MODE = "automatic"  # automatic or manual.
TOP_K = 10  # Number of ranked CoralSCOP candidates.
CORAL_COUNT = 3  # Number of corals in automatic mode.
SELECTED_CORALS = ()  # Manual ranks, e.g. (1, 3, 5); empty prompts.
OUTPUT_DIR = ROOT / "Output"  # Destination for figure folders.

MODEL_SIZE = 256
RESAMPLE = getattr(Image, "Resampling", Image)


def load_component(filename: str):
    """Input: component filename; return: imported module."""
    name = "coral_pipeline_" + Path(filename).stem.replace(" ", "_").lower()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def write_json(path: Path, data: Any) -> None:
    """Input: path and serializable data; save JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def find_environment(name: str, fallback: str | None = None) -> str:
    """Input: preferred and fallback environments; return: Python executable."""
    executable = "python.exe" if os.name == "nt" else "bin/python"
    roots = [Path(sys.prefix), Path(sys.base_prefix), Path.home() / "anaconda3",
             Path.home() / "miniconda3", Path.home() / "miniforge3"]
    if Path(sys.prefix).parent.name == "envs":
        roots.insert(0, Path(sys.prefix).parent.parent)
    for environment in (name, fallback):
        if environment is None:
            continue
        for base in roots:
            candidate = base / "envs" / environment / executable
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def run_worker(interpreter: str, operation: str, manifest: Path) -> None:
    """Input: executable, operation, and request path; run inference."""
    subprocess.run([interpreter, "-u", str(Path(__file__).resolve()), operation, str(manifest)], check=True)


def segment_worker(manifest: Path) -> None:
    """Input: segmentation request; save ranked candidate masks."""
    config = json.loads(manifest.read_text(encoding="utf-8"))
    import torch

    sys.path.insert(0, str(ROOT / "Models" / "CoralSCOP" / "architecture"))
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from segment_anything.utils.amg import rle_to_mask

    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CoralSCOP checkpoint not found: {checkpoint}")
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for CoralSCOP. Select a compatible Python or use --device cpu.")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model, points_per_side=config["points_per_side"], points_per_batch=config["points_per_batch"],
        pred_iou_thresh=config["pred_iou_thresh"], stability_score_thresh=config["stability_score_thresh"],
        crop_n_layers=config["crop_n_layers"], crop_n_points_downscale_factor=2,
        box_nms_thresh=0.60, crop_nms_thresh=0.60, output_mode="uncompressed_rle",
    )
    with Image.open(config["image"]) as source:
        rgb = np.asarray(source.convert("RGB"))
    with torch.inference_mode():
        proposals = generator.generate(rgb)
    proposals.sort(key=lambda item: (float(item["predicted_iou"]), float(item["stability_score"]),
                                     int(item["area"])), reverse=True)
    if not proposals:
        raise RuntimeError("CoralSCOP found no candidates. Try lower prediction thresholds.")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for rank, proposal in enumerate(proposals[:config["top_k"]], 1):
        mask = rle_to_mask(proposal["segmentation"])
        mask_path = output / f"candidate_{rank:02d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        candidates.append({"rank": rank, "mask": str(mask_path),
                           "predicted_iou": float(proposal["predicted_iou"]),
                           "stability_score": float(proposal["stability_score"]),
                           "area_pixels": int(np.count_nonzero(mask))})
    write_json(output / "candidates.json", {"candidates": candidates})


def dehaze_worker(manifest: Path) -> None:
    """Input: enhancement request; save enhanced model inputs."""
    config = json.loads(manifest.read_text(encoding="utf-8"))
    dehazer = load_component("Image Dehazer.py").ImageDehazer(**config["model"])
    for pair in config["images"]:
        with Image.open(pair["input"]) as image:
            enhanced = dehazer.enhance(image.convert("RGB"))
        enhanced.save(pair["output"])


def candidate_figure(image: Image.Image, candidates: list[dict], destination: Path, show: bool = True):
    """Input: source and ranked masks; return: displayed figure or None."""
    import matplotlib.pyplot as plt

    columns = min(5, len(candidates))
    rows = math.ceil(len(candidates) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.8 * rows), squeeze=False)
    thumbnail = image.copy()
    thumbnail.thumbnail((480, 480))
    base = np.asarray(thumbnail, dtype=np.float32)
    for axis, candidate in zip(axes.flat, candidates):
        with Image.open(candidate["mask"]) as mask_image:
            small_mask = np.asarray(mask_image.resize(thumbnail.size, RESAMPLE.NEAREST)) > 0
        overlay = base.copy()
        overlay[small_mask] = 0.55 * overlay[small_mask] + 0.45 * np.array([0, 255, 180])
        axis.imshow(overlay.astype(np.uint8))
        axis.set_title(f"Rank {candidate['rank']} | predicted IoU {candidate['predicted_iou']:.3f}")
        axis.axis("off")
    for axis in list(axes.flat)[len(candidates):]:
        axis.axis("off")
    figure.suptitle("Select coral ranks in the terminal")
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    if show:
        plt.show(block=False)
        plt.pause(0.1)
        return figure
    plt.close(figure)
    return None


def positive_integer(value: Any, name: str) -> int:
    """Input: value and name; return: validated positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def select_candidates(candidates: list[dict], analysis_mode: str, coral_count: int,
                      selections=None) -> list[dict]:
    """Input: ranked candidates and selection settings; return: chosen corals."""
    if not candidates:
        raise ValueError("No coral candidates are available.")
    if analysis_mode not in {"automatic", "manual"}:
        raise ValueError("Analysis mode must be automatic or manual.")
    if analysis_mode == "automatic":
        positive_integer(coral_count, "Coral count")
        if selections:
            raise ValueError("Explicit coral ranks require manual mode.")
        count = min(coral_count, len(candidates))
        if count < coral_count:
            print(f"Only {count} candidates are available; analyzing all {count}.", flush=True)
        return candidates[:count]

    def validate(ranks):
        """Input: distinct candidate ranks; return: selected coral records."""
        chosen = [positive_integer(rank, "Coral rank") for rank in ranks]
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError("Choose one or more distinct coral ranks.")
        lookup = {candidate["rank"]: candidate for candidate in candidates}
        if any(rank not in lookup for rank in chosen):
            raise ValueError(f"Choose ranks between 1 and {len(candidates)}.")
        return [lookup[rank] for rank in chosen]

    if selections:
        return validate(selections)
    while True:
        try:
            answer = input(f"Select coral ranks, separated by commas (1-{len(candidates)}), or q: ").strip()
        except EOFError as exc:
            raise RuntimeError("Terminal input is unavailable. Set SELECTED_CORALS or use --selection with manual mode.") from exc
        if answer.lower() == "q":
            raise SystemExit("Analysis cancelled.")
        try:
            ranks = [int(value) for value in answer.replace(",", " ").split()]
            return validate(ranks)
        except ValueError as exc:
            print(str(exc))


def square_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Input: binary mask; return: enclosing square (left, top, right, bottom)."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("The coral mask is empty.")
    left, top, right, bottom = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1
    side = max(right - left, bottom - top)
    x, y = (left + right - side) // 2, (top + bottom - side) // 2
    height, width = mask.shape
    if side <= width:
        x = min(max(0, x), width - side)
    if side <= height:
        y = min(max(0, y), height - side)
    return x, y, x + side, y + side


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Input: mask and target (width, height); return: aligned binary mask."""
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, RESAMPLE.NEAREST)) > 0
    if not np.any(resized):
        raise ValueError("A selected coral disappears after resizing. Use coral-resize or patches.")
    return resized


def prepare_inputs(image: Image.Image, mask: np.ndarray, mode: str,
                   stride: int | None = None, grid=None, num_patches=None,
                   coral_background: bool = False, patch_size: int = 256):
    """Input: RGB image, mask, and strategy; return: inputs, image, mask, geometry, batch."""
    if mode not in {"full-resize", "coral-resize", "patches", "overlap"}:
        raise ValueError(f"Unknown preparation strategy: {mode}")
    if mode in {"full-resize", "coral-resize"}:
        box = None
        if mode == "coral-resize":
            box = square_mask_bbox(mask)
            image = image.crop(box)
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).crop(box)) > 0
            if not coral_background:
                image = Image.fromarray(np.where(mask[..., None], np.asarray(image), 0).astype(np.uint8))
        image = image.resize((MODEL_SIZE, MODEL_SIZE), RESAMPLE.BICUBIC)
        aligned_mask = resize_mask(mask, image.size)
        return [image], image, aligned_mask, {"crop_box_xyxy": box}, None
    patcher = load_component("Image Patcher.py").ImagePatcher(
        patch_size=patch_size, overlap=mode == "overlap", stride=stride,
        grid=tuple(grid) if grid else None, num_patches=num_patches)
    batch = patcher.patch(image)
    return batch.patches, image, mask, batch.to_dict(), batch


def enhance_inputs(images: list[Image.Image], model_options: dict, interpreter: str,
                   work: Path) -> list[Image.Image]:
    """Input: prepared images and model settings; return: enhanced 256-pixel images."""
    work.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, image in enumerate(images):
        input_path, output_path = work / f"input_{index:06d}.png", work / f"output_{index:06d}.png"
        image.save(input_path)
        pairs.append({"input": str(input_path), "output": str(output_path)})
    images.clear()
    manifest = work / "request.json"
    write_json(manifest, {"model": model_options, "images": pairs})
    run_worker(interpreter, "_dehaze", manifest)
    predictions = []
    for pair in pairs:
        with Image.open(pair["output"]) as image:
            predictions.append(image.convert("RGB"))
    return predictions


def create_output_directory(base: Path, name: str) -> Path:
    """Input: destination and run name; return: a new output folder."""
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = base / (name if index == 1 else f"{name}_{index}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def build_parser() -> argparse.ArgumentParser:
    """Return: command-line options with defaults from the settings above."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=INPUT_IMAGE)
    parser.add_argument("--mode", "--strategy", choices=("full-resize", "coral-resize", "patches", "overlap"), default=PREPARATION_STRATEGY)
    parser.add_argument("--analysis-mode", choices=("automatic", "manual"), default=ANALYSIS_MODE)
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of ranked candidates")
    parser.add_argument("--coral-count", type=int, default=CORAL_COUNT, help="Number of corals in automatic mode")
    parser.add_argument("--selection", "--selections", nargs="+", type=int, default=list(SELECTED_CORALS), help="Manual coral ranks, separated by spaces")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, help="Overlap stride; defaults to PATCH_STRIDE")
    parser.add_argument("--model", default=DEHAZING_MODEL, help="Enhancement model listed in Models/registry.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mask", type=Path, help="Analyze one existing binary mask")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--grid", type=int, nargs=2, metavar=("COLS", "ROWS"), help="Patch grid; defaults to PATCH_GRID")
    layout.add_argument("--num-patches", type=int)
    parser.add_argument("--coral-background", action="store_true", help="Keep background in the coral square")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--segmentation-python", help="CoralSCOP Python executable")
    parser.add_argument("--model-python", help="Enhancement Python executable")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "Models/CoralSCOP/weights/CoralSCOP.pth")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Sampling steps; DCD-UIE and CLIP-UIE default to 2000")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--distance", choices=("rgb_euclidean", "cielab_euclidean", "ciede2000"), default="rgb_euclidean")
    parser.add_argument("--color-sampling", choices=("legacy-grid", "pixels"), default="legacy-grid")
    parser.add_argument("--color-block-size", type=int, default=1)
    parser.add_argument("--color-resize-dim", type=int)
    parser.add_argument("--lab-conversion", choices=("legacy", "srgb"), default="legacy")
    parser.add_argument("--no-show", action="store_true", help="View manual candidates using the printed preview path")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.40)
    parser.add_argument("--stability-score-thresh", type=float, default=0.50)
    return parser


def validate_options(args: argparse.Namespace):
    """Input: run settings; return: model key, stride, and validated adapter."""
    positive_integer(args.top_k, "Top k")
    if args.analysis_mode == "automatic":
        positive_integer(args.coral_count, "Coral count")
    positive_integer(args.patch_size, "Patch size")
    if args.analysis_mode == "automatic" and args.selection:
        raise ValueError("Use manual mode with --selection.")
    if args.selection:
        ranks = [positive_integer(value, "Coral rank") for value in args.selection]
        if len(ranks) != len(set(ranks)) or max(ranks) > args.top_k:
            raise ValueError("Manual ranks must be distinct and within top k.")
    if args.mask and args.selection:
        raise ValueError("Use either an existing mask or candidate ranks.")
    if args.mode in {"full-resize", "coral-resize"} and any(
            value is not None for value in (args.grid, args.num_patches, args.stride)):
        raise ValueError("Grid, patch count, and stride apply to patch strategies.")
    stride = args.stride if args.stride is not None else PATCH_STRIDE if args.mode == "overlap" else None
    if args.mode in {"patches", "overlap"}:
        if args.grid is None and args.num_patches is None:
            args.grid = PATCH_GRID
        load_component("Image Patcher.py").ImagePatcher(
            patch_size=args.patch_size, overlap=args.mode == "overlap", stride=stride,
            grid=tuple(args.grid) if args.grid else None, num_patches=args.num_patches)
    if args.points_per_side < 1 or args.points_per_batch < 1 or args.crop_n_layers < 0:
        raise ValueError("Sampling counts must be positive and crop layers nonnegative.")
    if not (0 <= args.pred_iou_thresh <= 1 and 0 <= args.stability_score_thresh <= 1):
        raise ValueError("Prediction thresholds must be between 0 and 1.")
    adapter = load_component("Image Dehazer.py").ImageDehazer(
        args.model, device=args.device, repo_root=args.repo_root, weights_path=args.weights,
        diffusion_steps=args.diffusion_steps, seed=args.seed, deterministic=args.deterministic)
    missing = [str(path) for path in adapter.required_weight_paths().values() if not path.is_file()]
    if not adapter.repo_root.is_dir() or missing:
        raise FileNotFoundError(f"Enhancement model files not found: {', '.join(missing) or adapter.repo_root}. "
                                "Follow docs/MODELS.md for the required model files.")
    return adapter.model_name, stride, adapter


def get_candidates(args: argparse.Namespace, work: Path) -> list[dict]:
    """Input: segmentation settings and temporary folder; return: ranked masks."""
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CoralSCOP checkpoint not found: {args.checkpoint}")
    destination = work / "segmentation"
    destination.mkdir()
    config = {"image": str(args.image.resolve()), "output": str(destination),
              "checkpoint": str(args.checkpoint.resolve()), "device": args.device, "top_k": args.top_k,
              "points_per_side": args.points_per_side, "points_per_batch": args.points_per_batch,
              "crop_n_layers": args.crop_n_layers, "pred_iou_thresh": args.pred_iou_thresh,
              "stability_score_thresh": args.stability_score_thresh}
    manifest = work / "segmentation_request.json"
    write_json(manifest, config)
    print("Finding coral candidates...", flush=True)
    run_worker(args.segmentation_python or find_environment("sam3"), "_segment", manifest)
    return json.loads((destination / "candidates.json").read_text(encoding="utf-8"))["candidates"]


def run_pipeline(args: argparse.Namespace) -> dict:
    """Input: run settings; return: output folder, figure paths, and coral results."""
    args.image = Path(args.image).resolve()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}. Set INPUT_IMAGE or provide a path.")
    model_key, stride, adapter = validate_options(args)
    matcher = load_component("Color Matcher.py").ColorMatcher(
        distance_mode=args.distance, sampling=args.color_sampling.replace("-", "_"),
        image_grid_size=args.color_block_size, resize_dim=args.color_resize_dim, lab_conversion=args.lab_conversion)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    environments = {"mulagan": "MulaGAN", "semiuir": "Semi-UIR", "dcduie": "DCD_UIE",
                    "dgdcgan": "DGD_cGAN", "clipuie": "CLIP_UIE"}
    interpreter = args.model_python or find_environment(
        environments.get(model_key, adapter.definition["directory"]),
        fallback="MulaGAN" if model_key == "clipuie" else None)
    model_options = {"model_name": args.model, "device": args.device,
                     "repo_root": str(adapter.repo_root), "weights_path": str(adapter.weights_path),
                     "diffusion_steps": args.diffusion_steps, "seed": args.seed, "deterministic": args.deterministic}

    with tempfile.TemporaryDirectory(prefix="coral_analysis_") as temporary:
        work = Path(temporary)
        if args.mask:
            selected = [{"rank": 1, "mask": str(args.mask.resolve()), "predicted_iou": None}]
        else:
            candidates = get_candidates(args, work)
            preview = None
            if args.analysis_mode == "manual":
                preview_path = work / "candidates.png"
                preview = candidate_figure(image, candidates, preview_path, show=not args.no_show)
                print(f"Candidate preview: {preview_path}", flush=True)
            try:
                selected = select_candidates(candidates, args.analysis_mode, args.coral_count, args.selection)
            finally:
                if preview is not None:
                    import matplotlib.pyplot as plt

                    plt.close(preview)
        masks = []
        for candidate in selected:
            with Image.open(candidate["mask"]) as source:
                if source.size != image.size:
                    raise ValueError("Coral mask dimensions must match the source image.")
                mask = np.asarray(source.convert("L")) > 0
            if not np.any(mask):
                raise ValueError(f"Candidate {candidate['rank']} has an empty mask.")
            masks.append(mask)
        print(f"Analyzing {len(selected)} coral(s) with {args.model}...", flush=True)
        if args.mode == "coral-resize":
            prepared = [prepare_inputs(image, mask, args.mode, coral_background=args.coral_background) for mask in masks]
            enhanced_images = enhance_inputs([item[0][0] for item in prepared], model_options, interpreter, work / "enhancement")
            analysis_masks = [item[2] for item in prepared]
        else:
            inputs, _, _, geometry, batch = prepare_inputs(
                image, masks[0], args.mode, stride, args.grid, args.num_patches,
                args.coral_background, args.patch_size)
            enhanced = enhance_inputs(inputs, model_options, interpreter, work / "enhancement")
            if batch is not None:
                patches = [patch.resize((args.patch_size, args.patch_size), RESAMPLE.BICUBIC)
                           if patch.size != (args.patch_size, args.patch_size) else patch for patch in enhanced]
                enhanced_image = load_component("Image Stitcher.py").ImageStitcher().stitch(patches, geometry)
            else:
                enhanced_image = enhanced[0]
            enhanced_images = [enhanced_image] * len(selected)
            analysis_masks = [resize_mask(mask, enhanced_image.size) for mask in masks]

        staging = work / "figures"
        staging.mkdir()
        results = []
        for candidate, source_mask, enhanced, analysis_mask in zip(selected, masks, enhanced_images, analysis_masks):
            result = matcher.analyze(enhanced, analysis_mask)
            name = f"coral_{candidate['rank']:02d}.png"
            matcher.save_analysis_figure(image, enhanced, analysis_mask, staging / name, result=result,
                                         source_mask=source_mask)
            results.append({"rank": candidate["rank"], "predicted_iou": candidate.get("predicted_iou"),
                            "bleaching_percentage": result["bleaching_percentage"],
                            "dominant_class": result["dominant_class"], "filename": name})
            print(f"Coral {candidate['rank']}: {result['bleaching_percentage']:.2f}% bleaching", flush=True)
        output = create_output_directory(Path(args.output).resolve(),
                                         f"{args.image.stem}_{model_key}_{args.mode}_{args.analysis_mode}")
        for result in results:
            destination = output / result.pop("filename")
            shutil.copy2(staging / destination.name, destination)
            result["figure"] = str(destination)
    print(f"Saved {len(results)} figure(s): {output}", flush=True)
    return {"output": str(output), "figures": [item["figure"] for item in results],
            "results": results, "selected_ranks": [item["rank"] for item in results],
            "mode": args.mode, "analysis_mode": args.analysis_mode}


def main() -> None:
    """Read settings, run analysis, and print the figure folder."""
    if len(sys.argv) > 1 and sys.argv[1] in {"_segment", "_dehaze"}:
        {"_segment": segment_worker, "_dehaze": dehaze_worker}[sys.argv[1]](Path(sys.argv[2]))
        return
    parser = build_parser()
    args = parser.parse_args()
    if args.image is None:
        parser.error("Set INPUT_IMAGE or provide an image path.")
    try:
        run_pipeline(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
