"""ImagePatcher splits RGB images into overlapping or non-overlapping patches, pads image edges, and stores the geometry needed to reconstruct the original image."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from PIL import Image


def _positive_integer(value: Any, name: str) -> int:
    """Validate a value and return a positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _rgb_image(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert a PIL image or uint8 RGB array to an RGB PIL image."""
    if isinstance(image, Image.Image):
        if image.width <= 0 or image.height <= 0:
            raise ValueError("The image must have nonzero width and height.")
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Image arrays must have shape (height, width, 3) and dtype uint8 (RGB).")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("The image must have nonzero width and height.")
    return Image.fromarray(array, "RGB")


@dataclass
class PatchBatch:
    patches: list[Image.Image]
    positions: list[tuple[int, int]]
    original_size: tuple[int, int]
    working_size: tuple[int, int]
    padded_size: tuple[int, int]
    patch_size: int
    stride: int
    grid: tuple[int, int]

    @property
    def num_patches(self) -> int:
        """Return the number of stored patches."""
        return len(self.patches)

    def __len__(self) -> int:
        """Return the number of stored patches."""
        return self.num_patches

    def to_dict(self) -> dict[str, Any]:
        """Return reconstruction geometry as a JSON-compatible dictionary."""
        return {
            "schema_version": 1,
            "original_size": list(self.original_size),
            "working_size": list(self.working_size),
            "padded_size": list(self.padded_size),
            "patch_size": self.patch_size,
            "stride": self.stride,
            "grid": list(self.grid),
            "positions": [list(position) for position in self.positions],
            "num_patches": self.num_patches,
        }


class ImagePatcher:
    def __init__(
        self,
        patch_size: int = 256,  # Patch width and height in pixels.
        overlap: bool = False,  # Enable overlapping patches.
        stride: int | None = None,  # Patch spacing; None uses the default.
        grid: tuple[int, int] | None = None,  # Optional (columns, rows) grid.
        num_patches: int | None = None,  # Optional exact patch count.
    ) -> None:
        """Configure patch dimensions and layout; returns None."""
        self.patch_size = _positive_integer(patch_size, "patch_size")
        if not isinstance(overlap, bool):
            raise ValueError("overlap must be True or False.")
        self.overlap = overlap
        default_stride = max(1, self.patch_size // 2) if overlap else self.patch_size
        self.stride = _positive_integer(default_stride if stride is None else stride, "stride")
        if overlap and not self.stride < self.patch_size:
            raise ValueError("Overlapping patches require stride < patch_size.")
        if not overlap and self.stride != self.patch_size:
            raise ValueError("Non-overlapping patches require stride == patch_size.")
        if grid is not None and num_patches is not None:
            raise ValueError("Specify grid or num_patches, not both.")
        if grid is not None:
            if not isinstance(grid, (tuple, list)) or len(grid) != 2:
                raise ValueError("grid must be (columns, rows).")
            grid = tuple(_positive_integer(value, "grid dimension") for value in grid)
        self.grid = grid
        self.num_patches = None if num_patches is None else _positive_integer(num_patches, "num_patches")

    def _count_grid(self, width: int, height: int) -> tuple[int, int]:
        """Take image width and height; return the closest matching grid."""
        count = self.num_patches
        assert count is not None
        candidates = [(columns, count // columns) for columns in range(1, count + 1) if count % columns == 0]

        def aspect_error(grid: tuple[int, int]) -> float:
            """Take a grid; return its aspect-ratio error."""
            columns, rows = grid
            canvas_width = self.patch_size + (columns - 1) * self.stride
            canvas_height = self.patch_size + (rows - 1) * self.stride
            return abs(np.log((canvas_width / canvas_height) / (width / height)))

        return min(candidates, key=aspect_error)

    def patch(self, image: Image.Image | np.ndarray) -> PatchBatch:
        """Take a PIL image or RGB array; return patches and geometry as PatchBatch."""
        source = _rgb_image(image)
        original_size = source.size
        width, height = original_size
        grid = self.grid
        if grid is None and self.num_patches is not None:
            grid = self._count_grid(width, height)

        if grid is not None:
            columns, rows = grid
            working_size = (
                self.patch_size + (columns - 1) * self.stride,
                self.patch_size + (rows - 1) * self.stride,
            )
            source = source.resize(working_size, Image.Resampling.BICUBIC)
        else:
            columns = max(1, ceil(max(0, width - self.patch_size) / self.stride) + 1)
            rows = max(1, ceil(max(0, height - self.patch_size) / self.stride) + 1)
            grid = (columns, rows)
            working_size = original_size

        padded_size = (
            self.patch_size + (columns - 1) * self.stride,
            self.patch_size + (rows - 1) * self.stride,
        )
        array = np.asarray(source)
        array = np.pad(
            array,
            ((0, padded_size[1] - source.height), (0, padded_size[0] - source.width), (0, 0)),
            mode="edge",
        )
        positions = [(column * self.stride, row * self.stride) for row in range(rows) for column in range(columns)]
        patches = [
            Image.fromarray(array[y:y + self.patch_size, x:x + self.patch_size].copy(), "RGB")
            for x, y in positions
        ]
        return PatchBatch(
            patches=patches,
            positions=positions,
            original_size=original_size,
            working_size=working_size,
            padded_size=padded_size,
            patch_size=self.patch_size,
            stride=self.stride,
            grid=grid,
        )
