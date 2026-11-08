"""Match a selected coral to the CoralWatch palette, estimate bleaching from its shade grades, and save comparison figures using RGB images and explicit binary masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


MANUAL_PALETTE = {
    "B1": (247, 248, 232), "B2": (244, 247, 190), "B3": (235, 241, 133),
    "B4": (202, 208, 58), "B5": (153, 159, 25), "B6": (101, 121, 0),
    "C1": (246, 236, 234), "C2": (243, 203, 193), "C3": (235, 158, 138),
    "C4": (201, 97, 68), "C5": (153, 61, 36), "C6": (116, 21, 1),
    "D1": (246, 236, 224), "D2": (244, 220, 192), "D3": (235, 189, 137),
    "D4": (207, 150, 83), "D5": (151, 94, 27), "D6": (115, 59, 0),
    "E1": (245, 242, 225), "E2": (245, 233, 191), "E3": (237, 214, 136),
    "E4": (207, 175, 72), "E5": (156, 128, 31), "E6": (116, 89, 0),
}

ImageInput = Union[Image.Image, np.ndarray]


def _rgb_array(image: ImageInput) -> np.ndarray:
    """PIL image or RGB uint8 array -> validated RGB array."""
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or not all(array.shape[:2]):
        raise ValueError("image must be a nonempty H x W x 3 RGB image.")
    if array.dtype != np.uint8:
        raise TypeError("RGB arrays must have dtype uint8 with values 0--255.")
    return array


def _binary_mask(mask: ImageInput, shape: Tuple[int, int]) -> np.ndarray:
    """Mask and image shape -> validated binary foreground array."""
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError("mask must be a 2D binary array matching the image's height and width.")
    if array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("mask must contain finite binary values (0/1 or 0/255).")
    values = np.unique(array)
    if not np.all(np.isin(values, [0, 1, 255])):
        raise ValueError("mask must contain binary values (0/1 or 0/255), not soft probabilities.")
    result = array > 0
    if not np.any(result):
        raise ValueError("The selected coral mask is empty.")
    return result


class ColorMatcher:
    DISTANCE_MODES = ("rgb_euclidean", "cielab_euclidean", "ciede2000")
    SERIES = ("B", "C", "D", "E")

    def __init__(
        self,
        distance_mode: str = "rgb_euclidean",  # Color distance.
        bleached_grades: Sequence[int] = (1, 2, 3),  # Bleached shades.
        chunk_size: int = 65536,  # Samples per batch.
        lab_conversion: str = "legacy",  # legacy or srgb.
        sampling: str = "legacy_grid",  # legacy_grid or pixels.
        resize_dim: Optional[int] = None,  # Longest side; None keeps size.
        image_grid_size: float = 1,  # Grid block size.
    ) -> None:
        """Matching settings -> configured CoralWatch matcher."""
        self.distance_mode = str(distance_mode).lower()
        if self.distance_mode not in self.DISTANCE_MODES:
            raise ValueError(f"distance_mode must be one of {self.DISTANCE_MODES}.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        grades = tuple(bleached_grades)
        if not grades or any(isinstance(g, bool) or not isinstance(g, (int, np.integer)) or g not in range(1, 7) for g in grades):
            raise ValueError("bleached_grades must contain integers between 1 and 6.")
        self.bleached_grades = tuple(sorted(set(int(g) for g in grades)))
        if lab_conversion not in ("legacy", "srgb"):
            raise ValueError("lab_conversion must be 'legacy' or 'srgb'.")
        self.lab_conversion = lab_conversion
        if sampling not in ("pixels", "legacy_grid"):
            raise ValueError("sampling must be 'pixels' or 'legacy_grid'.")
        if resize_dim is not None and (isinstance(resize_dim, bool) or not isinstance(resize_dim, int) or resize_dim <= 0):
            raise ValueError("resize_dim must be a positive integer or None.")
        if isinstance(image_grid_size, bool) or not np.isfinite(image_grid_size) or image_grid_size <= 0:
            raise ValueError("image_grid_size must be positive.")
        self.sampling = sampling
        self.resize_dim = resize_dim
        self.image_grid_size = float(image_grid_size)
        self.chunk_size = chunk_size
        self.palette = dict(MANUAL_PALETTE)
        self.labels = list(self.palette)
        self.reference_rgb = np.asarray(list(self.palette.values()), dtype=np.float32)
        self._colour = None
        if self.distance_mode != "rgb_euclidean":
            try:
                import colour
            except ImportError as exc:
                raise ImportError("Lab matching requires colour-science: python -m pip install colour-science") from exc
            self._colour = colour
            self.reference_values = self._to_lab(self.reference_rgb)

            if self.distance_mode == "cielab_euclidean":
                self.reference_values = self.reference_values.astype(np.float32)
        else:
            self.reference_values = self.reference_rgb

    def _to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """RGB samples -> Lab samples using the selected conversion."""
        colourspace = self._colour.RGB_COLOURSPACES["sRGB"]
        xyz = self._colour.RGB_to_XYZ(
            np.asarray(rgb, dtype=np.float64) / 255.0,
            colourspace=colourspace,
            apply_cctf_decoding=self.lab_conversion == "srgb",
        )
        return self._colour.XYZ_to_Lab(xyz, colourspace.whitepoint)

    def _nearest(self, rgb: np.ndarray, references: np.ndarray) -> np.ndarray:
        """RGB samples and reference colors -> nearest reference indices."""
        if self.distance_mode == "rgb_euclidean":
            values = rgb.astype(np.float32)
        else:
            values = self._to_lab(rgb)
        if self.distance_mode == "ciede2000":
            distances = self._colour.difference.delta_E(
                values[:, None, :], references[None, :, :], method="CIE 2000"
            )
        else:
            delta = values[:, None, :] - references[None, :, :]
            distances = np.sum(delta * delta, axis=2)
        return np.argmin(distances, axis=1).astype(np.uint8)

    def _sample(self, rgb: np.ndarray, foreground: np.ndarray) -> tuple:
        """RGB image and foreground mask -> samples and sampling geometry."""
        height, width = foreground.shape
        sampled_rgb, sampled_mask = rgb, foreground
        if self.resize_dim is not None:
            import cv2

            scale = self.resize_dim / float(max(height, width))
            resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

            sampled_rgb = cv2.resize(np.where(foreground[..., None], rgb, 0), resized_size, interpolation=cv2.INTER_AREA)
            sampled_mask = cv2.resize(foreground.astype(np.float32), resized_size, interpolation=cv2.INTER_AREA) > 0
        sampled_h, sampled_w = sampled_mask.shape
        geometry = {
            "resized_image_size": [sampled_w, sampled_h],
            "resize_dim": self.resize_dim,
            "image_grid_size": self.image_grid_size if self.sampling == "legacy_grid" else None,
            "mask_rule": "explicit binary mask; true black coral is included",
        }
        if self.sampling == "pixels":
            geometry.update({"grid_dimensions": None, "cell_range": None})
            return sampled_rgb[sampled_mask], sampled_mask, (0, 0), sampled_mask.shape, geometry

        target_max_side = self.resize_dim if self.resize_dim is not None else max(height, width)
        requested_grid = max(1, int(target_max_side / self.image_grid_size))
        grid = min(requested_grid, sampled_h, sampled_w)
        y_edges = np.linspace(0, sampled_h, grid + 1, dtype=np.int32)
        x_edges = np.linspace(0, sampled_w, grid + 1, dtype=np.int32)
        ys, xs = np.nonzero(sampled_mask)
        row0 = int(np.searchsorted(y_edges, ys.min(), side="right") - 1)
        row1 = int(np.searchsorted(y_edges, ys.max(), side="right"))
        col0 = int(np.searchsorted(x_edges, xs.min(), side="right") - 1)
        col1 = int(np.searchsorted(x_edges, xs.max(), side="right"))
        y0, y1, x0, x1 = y_edges[row0], y_edges[row1], x_edges[col0], x_edges[col1]
        local_mask = sampled_mask[y0:y1, x0:x1]
        local_rgb = sampled_rgb[y0:y1, x0:x1]
        local_y = y_edges[row0:row1] - y0
        local_x = x_edges[col0:col1] - x0

        def reduce_cells(values: np.ndarray, dtype) -> np.ndarray:
            """Array and accumulation dtype -> cell sums."""
            return np.add.reduceat(np.add.reduceat(values, local_y, axis=0, dtype=dtype), local_x, axis=1, dtype=dtype)

        counts = reduce_cells(local_mask, np.uint64)
        valid_cells = counts > 0
        inverse_counts = 1.0 / counts[valid_cells]
        means = np.empty((int(valid_cells.sum()), 3), dtype=np.uint8)
        for channel in range(3):
            sums = reduce_cells(np.where(local_mask, local_rgb[..., channel], 0), np.uint64)

            means[:, channel] = (sums[valid_cells] * inverse_counts).astype(np.uint8)
        geometry.update({
            "grid_dimensions": [grid, grid],
            "requested_grid_size": requested_grid,
            "cell_edges": "floor(linspace(0, image_dimension, grid+1))",
            "cell_range": {"rows": [row0, row1], "columns": [col0, col1]},
            "grade_map_expansion": "actual floor-linspace cell edges; nearest resize if needed; clipped to explicit mask",
        })
        return means, valid_cells, (row0, col0), (grid, grid), geometry

    def analyze_with_map(self, image: ImageInput, mask: ImageInput) -> Tuple[Dict[str, Any], np.ndarray]:
        """RGB image and mask -> statistics and aligned grade map."""
        rgb = _rgb_array(image)
        foreground = _binary_mask(mask, rgb.shape[:2])
        pixels, sample_mask, sample_origin, canvas_shape, geometry = self._sample(rgb, foreground)
        count = len(pixels)
        initial = np.empty(count, dtype=np.uint8)
        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            initial[start:end] = self._nearest(pixels[start:end], self.reference_values)

        first_counts = np.bincount(initial, minlength=24).reshape(4, 6)
        scores = first_counts[:, 2:6].sum(axis=1)
        dominant_index = int(np.argmax(scores))
        dominant_class = self.SERIES[dominant_index]
        selected_refs = self.reference_values[dominant_index * 6:(dominant_index + 1) * 6]
        grades = np.empty(count, dtype=np.uint8)
        for start in range(0, count, self.chunk_size):
            end = min(start + self.chunk_size, count)
            grades[start:end] = self._nearest(pixels[start:end], selected_refs) + 1

        grade_counts = np.bincount(grades, minlength=7)[1:7]
        bleached = sum(int(grade_counts[g - 1]) for g in self.bleached_grades)
        per_series = {}
        for index, series in enumerate(self.SERIES):
            series_count = int(first_counts[index].sum())
            series_bleached = sum(int(first_counts[index, g - 1]) for g in self.bleached_grades)
            per_series[series] = {
                "initial_matched_samples": series_count,
                "initial_grade_counts": {str(g): int(first_counts[index, g - 1]) for g in range(1, 7)},
                "dominant_series_score": int(scores[index]),
                "initial_bleached_samples": series_bleached,
                "initial_bleaching_percentage": 100.0 * series_bleached / series_count if series_count else None,
            }

        report = {
            "distance_mode": self.distance_mode,
            "lab_conversion": self.lab_conversion if self.distance_mode != "rgb_euclidean" else None,
            "color_order": "RGB",
            "assignment_method": "closest of 24 colors; dominant series by grades 3-6; closest of its 6 shades",
            "sampling": self.sampling,
            "sampling_geometry": geometry,
            "count_unit": "cells" if self.sampling == "legacy_grid" else "pixels",
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "foreground_pixels": int(foreground.sum()),
            "total_samples": count,
            "dominant_class": dominant_class,
            "dominant_series_tie_order": list(self.SERIES),
            "bleached_grades": list(self.bleached_grades),
            "bleaching_weights": {str(g): int(g in self.bleached_grades) for g in range(1, 7)},
            "bleached_samples": bleached,
            "healthy_samples": count - bleached,
            "bleached_pixels": bleached if self.sampling == "pixels" else None,
            "healthy_pixels": count - bleached if self.sampling == "pixels" else None,
            "bleaching_percentage": 100.0 * bleached / count,
            "grade_counts": {str(g): int(grade_counts[g - 1]) for g in range(1, 7)},
            "grade_percentages": {str(g): 100.0 * int(grade_counts[g - 1]) / count for g in range(1, 7)},
            "per_series": per_series,
            "palette_rgb": {name: list(color) for name, color in self.palette.items()},
        }
        grade_map = np.zeros(canvas_shape, dtype=np.uint8)
        row0, col0 = sample_origin
        region = grade_map[row0:row0 + sample_mask.shape[0], col0:col0 + sample_mask.shape[1]]
        region[sample_mask] = grades
        if self.sampling == "legacy_grid":

            sampled_w, sampled_h = geometry["resized_image_size"]
            y_edges = np.linspace(0, sampled_h, canvas_shape[0] + 1, dtype=np.int32)
            x_edges = np.linspace(0, sampled_w, canvas_shape[1] + 1, dtype=np.int32)
            cell_rows = np.searchsorted(y_edges, np.arange(sampled_h), side="right") - 1
            cell_cols = np.searchsorted(x_edges, np.arange(sampled_w), side="right") - 1
            grade_map = grade_map[cell_rows[:, None], cell_cols[None, :]]
        if grade_map.shape != rgb.shape[:2]:
            import cv2

            grade_map = cv2.resize(grade_map, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        grade_map[~foreground] = 0
        return report, grade_map

    def analyze(self, image: ImageInput, mask: ImageInput) -> Dict[str, Any]:
        """RGB image and mask -> color counts and bleaching statistics."""
        return self.analyze_with_map(image, mask)[0]

    def grade_map(self, image: ImageInput, mask: ImageInput, result: Optional[dict] = None) -> np.ndarray:
        """RGB image, mask, optional statistics -> aligned grade labels."""
        report, grades = self.analyze_with_map(image, mask)
        if result is not None:
            for key in ("distance_mode", "sampling", "dominant_class", "foreground_pixels", "grade_counts"):
                if report[key] != result.get(key):
                    raise ValueError(f"The supplied result does not describe this image/mask ({key}).")
        return grades

    def _palette_frame(self, color_map: np.ndarray, foreground: np.ndarray) -> Image.Image:
        """Matched RGB map and mask -> image with palette borders."""
        height, width = foreground.shape
        thickness = max(2, round(min(height, width) / 50))
        margin = max(2, thickness // 2)
        padding = thickness + margin
        canvas = np.pad(color_map, ((padding, padding), (padding, padding), (0, 0)))
        ys, xs = np.nonzero(foreground)
        left, right = int(xs.min()) + padding, int(xs.max()) + padding + 1
        top, bottom = int(ys.min()) + padding, int(ys.max()) + padding + 1
        x0, x1 = left - padding, right + padding
        y0, y1 = top - padding, bottom + padding

        def gradient(series: str, length: int, reverse: bool = False) -> np.ndarray:
            """Palette series and length -> interpolated RGB strip."""
            colors = np.array([self.palette[f"{series}{g}"] for g in range(1, 7)], dtype=np.float64)
            if reverse:
                colors = colors[::-1]
            positions = np.linspace(0, 5, length)
            return np.stack([np.interp(positions, np.arange(6), colors[:, c]) for c in range(3)], axis=-1).astype(np.uint8)

        canvas[y0:y0 + thickness, x0:x1] = gradient("B", x1 - x0)[None, :, :]
        canvas[y1 - thickness:y1, x0:x1] = gradient("D", x1 - x0, True)[None, :, :]
        canvas[y0:y1, x0:x0 + thickness] = gradient("E", y1 - y0, True)[:, None, :]
        canvas[y0:y1, x1 - thickness:x1] = gradient("C", y1 - y0)[:, None, :]
        return Image.fromarray(canvas)

    def save_analysis_figure(
        self,
        input_image: ImageInput,
        enhanced: ImageInput,
        mask: ImageInput,
        output_path: Union[str, Path],
        result: Optional[Dict[str, Any]] = None,
        source_mask: Optional[ImageInput] = None,
    ) -> str:
        """RGB images and aligned masks -> one saved three-panel figure path."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch, Rectangle

        source_rgb, enhanced_rgb = _rgb_array(input_image), _rgb_array(enhanced)
        foreground = _binary_mask(mask, enhanced_rgb.shape[:2])
        report, grades = self.analyze_with_map(enhanced_rgb, foreground)
        if result is not None:
            for key in ("distance_mode", "sampling", "dominant_class", "foreground_pixels", "grade_counts"):
                if report[key] != result.get(key):
                    raise ValueError(f"The supplied result does not describe this image/mask ({key}).")
        original_mask = None
        if source_mask is not None:
            original_mask = _binary_mask(source_mask, source_rgb.shape[:2])
        elif source_rgb.shape[:2] == foreground.shape:
            original_mask = foreground

        context = Image.fromarray(source_rgb)
        context.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        context_array = np.asarray(context).copy()
        if original_mask is not None:
            context_mask = np.asarray(Image.fromarray(original_mask).resize(context.size, Image.Resampling.NEAREST))
            context_array[context_mask] = (
                0.65 * context_array[context_mask] + 0.35 * np.array((0, 230, 170))
            ).astype(np.uint8)

        ys, xs = np.nonzero(foreground)
        margin = max(2, round(max(np.ptp(ys) + 1, np.ptp(xs) + 1) * 0.05))
        y0, y1 = max(0, int(ys.min()) - margin), min(foreground.shape[0], int(ys.max()) + margin + 1)
        x0, x1 = max(0, int(xs.min()) - margin), min(foreground.shape[1], int(xs.max()) + margin + 1)
        crop_mask = foreground[y0:y1, x0:x1]
        segmented = Image.fromarray(np.where(crop_mask[..., None], enhanced_rgb[y0:y1, x0:x1], 0).astype(np.uint8))
        dominant = report["dominant_class"]
        colors = np.array([(0, 0, 0)] + [self.palette[f"{dominant}{grade}"] for grade in range(1, 7)], dtype=np.uint8)
        mapped = self._palette_frame(colors[grades[y0:y1, x0:x1]], crop_mask)
        segmented.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        mapped.thumbnail((1000, 1000), Image.Resampling.NEAREST)

        panel_height = min(8.0, max(5.5, 4.5 * max(
            context.height / context.width, segmented.height / segmented.width, mapped.height / mapped.width,
        )))
        figure = Figure(figsize=(15, panel_height + 1.5), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        for axis, displayed, panel_title in zip(
            axes, (context_array, segmented, mapped),
            ("Input image", "Selected coral (enhanced)",
             f"CoralWatch {dominant}\n{report['bleaching_percentage']:.2f}% bleaching"),
        ):
            axis.imshow(displayed)
            axis.set_title(panel_title)
            axis.set_axis_off()
        if original_mask is not None:
            source_y, source_x = np.nonzero(original_mask)
            scale_x, scale_y = context.width / source_rgb.shape[1], context.height / source_rgb.shape[0]
            axes[0].add_patch(Rectangle(
                (source_x.min() * scale_x, source_y.min() * scale_y),
                (np.ptp(source_x) + 1) * scale_x, (np.ptp(source_y) + 1) * scale_y,
                fill=False, edgecolor="#00d5a0", linewidth=1.5,
            ))
        axes[2].legend(
            handles=[Patch(facecolor=colors[grade] / 255.0, edgecolor="black", label=f"{dominant}{grade}")
                     for grade in range(1, 7)],
            loc="upper center", bbox_to_anchor=(0.5, 0), ncol=6, fontsize=12,
            borderaxespad=0.45, handlelength=1.3, handletextpad=0.45, columnspacing=0.75,
        )
        destination = Path(output_path)
        if destination.suffix.lower() != ".png":
            raise ValueError("output_path must have a .png extension.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.savefig(destination, dpi=150, bbox_inches="tight", pad_inches=0.2)
        finally:
            figure.clear()
        return str(destination.resolve())

    def save_visualizations(
        self,
        original: ImageInput,
        enhanced: ImageInput,
        mask: ImageInput,
        output_dir: Union[str, Path],
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Aligned RGB images and mask -> saved image and statistics paths."""
        original_rgb, enhanced_rgb = _rgb_array(original), _rgb_array(enhanced)
        if original_rgb.shape != enhanced_rgb.shape:
            raise ValueError("original and enhanced must have the same dimensions for visualization.")
        foreground = _binary_mask(mask, enhanced_rgb.shape[:2])
        report, grades = self.analyze_with_map(enhanced_rgb, foreground)
        if result is not None:
            for key in ("distance_mode", "sampling", "dominant_class", "foreground_pixels", "grade_counts"):
                if report[key] != result.get(key):
                    raise ValueError(f"The supplied result does not describe this image/mask ({key}).")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        original_masked = np.where(foreground[..., None], original_rgb, 0).astype(np.uint8)
        enhanced_masked = np.where(foreground[..., None], enhanced_rgb, 0).astype(np.uint8)
        dominant = report["dominant_class"]
        lut = np.array([(0, 0, 0)] + [self.palette[f"{dominant}{g}"] for g in range(1, 7)], dtype=np.uint8)
        mapped = lut[grades]
        framed = self._palette_frame(mapped, foreground)
        images = {
            "original_masked": Image.fromarray(original_masked),
            "enhanced_masked": Image.fromarray(enhanced_masked),
            "bleaching_mask": Image.fromarray(grades),
            "color_matched": Image.fromarray(mapped),
            "palette_framed": framed,
        }
        paths = {}
        for name, image in images.items():
            target = output / f"{name}.png"
            image.save(target)
            paths[name] = str(target.resolve())
        json_path = output / "color_statistics.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["statistics_json"] = str(json_path.resolve())
        csv_path = output / "color_statistics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["shade", report["count_unit"], "percentage", "bleaching_weight"])
            for grade in range(1, 7):
                writer.writerow([f"{dominant}{grade}", report["grade_counts"][str(grade)], report["grade_percentages"][str(grade)], int(grade in self.bleached_grades)])
        paths["statistics_csv"] = str(csv_path.resolve())

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch

        figure = Figure(figsize=(16, 6), constrained_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        ys, xs = np.nonzero(foreground)
        crop_margin = max(2, int(round(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 0.05)))
        crop_y0, crop_y1 = max(0, int(ys.min()) - crop_margin), min(foreground.shape[0], int(ys.max()) + crop_margin + 1)
        crop_x0, crop_x1 = max(0, int(xs.min()) - crop_margin), min(foreground.shape[1], int(xs.max()) + crop_margin + 1)
        cropped_mask = foreground[crop_y0:crop_y1, crop_x0:crop_x1]
        display_framed = self._palette_frame(mapped[crop_y0:crop_y1, crop_x0:crop_x1], cropped_mask)
        for axis, displayed, title in zip(
            axes,
            [Image.fromarray(original_masked[crop_y0:crop_y1, crop_x0:crop_x1]), Image.fromarray(enhanced_masked[crop_y0:crop_y1, crop_x0:crop_x1]), display_framed],
            ["Selected coral (original)", "Selected coral (enhanced)", f"CoralWatch {dominant}: {report['bleaching_percentage']:.2f}% bleaching"],
        ):
            displayed.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            axis.imshow(displayed)
            axis.set_title(title)
            axis.set_axis_off()
        axes[2].legend(
            handles=[Patch(facecolor=np.array(self.palette[f"{dominant}{g}"]) / 255.0, edgecolor="black", label=f"{dominant}{g}") for g in range(1, 7)],
            loc="upper right", framealpha=0.9,
        )
        figure_path = output / "visualization.png"
        figure.savefig(figure_path, dpi=150)
        figure.clear()
        paths["visualization"] = str(figure_path.resolve())
        return paths
