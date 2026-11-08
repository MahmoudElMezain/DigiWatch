"""ImageStitcher reconstructs an RGB image from patches and their geometry, averages overlapping pixels, removes padding, and restores the original image dimensions."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    """Validate a value and return an integer at or above minimum."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _size(value: Any, name: str) -> tuple[int, int]:
    """Validate dimensions and return a (width, height) tuple."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must contain (width, height).")
    return tuple(_integer(number, name) for number in value)


class ImageStitcher:
    def stitch(
        self,
        patches: Iterable[Image.Image | np.ndarray],
        metadata: Any,
    ) -> Image.Image:
        """Take RGB patches and PatchBatch or dictionary metadata; return an RGB PIL image."""
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a PatchBatch or geometry dictionary.")

        original_size = _size(metadata.get("original_size"), "original_size")
        working_size = _size(metadata.get("working_size"), "working_size")
        padded_size = _size(metadata.get("padded_size"), "padded_size")
        patch_size = _integer(metadata.get("patch_size"), "patch_size")
        if working_size[0] > padded_size[0] or working_size[1] > padded_size[1]:
            raise ValueError("working_size cannot exceed padded_size.")
        positions = metadata.get("positions")
        if not isinstance(positions, (tuple, list)) or not positions:
            raise ValueError("metadata must contain nonempty patch positions.")
        normalized_positions = []
        for position in positions:
            if not isinstance(position, (tuple, list)) or len(position) != 2:
                raise ValueError("Each patch position must contain (x, y).")
            x, y = (_integer(value, "patch coordinate", minimum=0) for value in position)
            if x + patch_size > padded_size[0] or y + patch_size > padded_size[1]:
                raise ValueError("A patch position extends outside padded_size.")
            normalized_positions.append((x, y))
        if len(set(normalized_positions)) != len(normalized_positions):
            raise ValueError("Duplicate patch positions are not allowed.")

        patches = list(patches)
        count = _integer(metadata.get("num_patches", len(positions)), "num_patches")
        if len(patches) != len(positions) or len(patches) != count:
            raise ValueError("Patch count must match the number of metadata positions and num_patches.")

        total = np.zeros((padded_size[1], padded_size[0], 3), dtype=np.float32)
        coverage = np.zeros((padded_size[1], padded_size[0]), dtype=np.uint32)
        for patch, (x, y) in zip(patches, normalized_positions):
            if isinstance(patch, Image.Image):
                array = np.asarray(patch.convert("RGB"))
            else:
                array = np.asarray(patch)
            if array.dtype != np.uint8 or array.shape != (patch_size, patch_size, 3):
                raise ValueError(f"Every patch must be uint8 RGB with shape ({patch_size}, {patch_size}, 3).")
            total[y:y + patch_size, x:x + patch_size] += array
            coverage[y:y + patch_size, x:x + patch_size] += 1
        if np.any(coverage == 0):
            raise ValueError("Patch metadata leaves uncovered pixels in the reconstruction canvas.")

        total /= coverage[:, :, None]
        reconstructed = np.rint(total[:working_size[1], :working_size[0]]).clip(0, 255).astype(np.uint8)
        output = Image.fromarray(reconstructed, "RGB")
        if output.size != original_size:
            output = output.resize(original_size, Image.Resampling.BICUBIC)
        return output
