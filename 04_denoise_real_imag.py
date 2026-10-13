"""Real/Imaginary split MP-PCA denoising.

The real and imaginary components are denoised independently with a
signed MP-PCA implementation that does not clip negative values.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat

# 절삭을 제거한 로컬 MP-PCA 구현
from localpca_dn import mppca


PROJECT_ROOT = Path(__file__).resolve().parent

ORIGINAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "meas_gre_dir1.mat"
)

NOISY_DIR = (
    PROJECT_ROOT
    / "results"
    / "noisy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "real_imag"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real/Imaginary split MP-PCA denoising"
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[50],
        help="Noise levels to process",
    )

    parser.add_argument(
        "--patch-radii",
        type=int,
        nargs="+",
        default=[2],
        help="MP-PCA patch radii",
    )

    parser.add_argument(
        "--slice",
        type=int,
        default=88,
        dest="slice_index",
    )

    parser.add_argument(
        "--echo",
        type=int,
        default=0,
        dest="echo_index",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing result",
    )

    return parser.parse_args()


def validate_result(
    denoised_real,
    denoised_imag,
    mask,
):
    real_finite = np.isfinite(
        denoised_real[mask]
    ).all()

    imag_finite = np.isfinite(
        denoised_imag[mask]
    ).all()

    if not real_finite:
        raise RuntimeError(
            "Real result contains NaN or Inf inside mask"
        )

    if not imag_finite:
        raise RuntimeError(
            "Imaginary result contains NaN or Inf inside mask"
        )

    real_negative = int(
        np.count_nonzero(
            denoised_real[mask] < 0
        )
    )

    imag_negative = int(
        np.count_nonzero(
            denoised_imag[mask] < 0
        )
    )

    if real_negative == 0:
        raise RuntimeError(
            "Real result has no negative values; "
            "nonnegative clipping may have occurred"
        )

    if imag_negative == 0:
        raise RuntimeError(
            "Imaginary result has no negative values; "
            "nonnegative clipping may have occurred"
        )

    print(
        f"Real negative voxel values: {real_negative:,}",
        flush=True,
    )

    print(
        f"Imaginary negative voxel values: {imag_negative:,}",
        flush=True,
    )

    print(
        "In-mask finite check: passed",
        flush=True,
    )


def save_preview(
    original,
    noisy_real,
    noisy_imag,
    denoised_real,
    denoised_imag,
    mask,
    output_path,
    level,
    patch_radius,
    slice_index,
    echo_index,
):
    original_slice = np.abs(
        original[
            :,
            :,
            slice_index,
            echo_index,
        ]
    )

    noisy_slice = np.hypot(
        noisy_real[
            :,
            :,
            slice_index,
            echo_index,
        ],
        noisy_imag[
            :,
            :,
            slice_index,
            echo_index,
        ],
    )

    denoised_slice = np.hypot(
        denoised_real[
            :,
            :,
            slice_index,
            echo_index,
        ],
        denoised_imag[
            :,
            :,
            slice_index,
            echo_index,
        ],
    )

    mask_slice = mask[
        :,
        :,
        slice_index,
    ]

    masked_original = original_slice[
        mask_slice
    ]

    vmin, vmax = np.percentile(
        masked_original,
        [1, 99],
    )

    difference = (
        denoised_slice
        - original_slice
    )

    difference_limit = float(
        np.percentile(
            np.abs(
                difference[mask_slice]
            ),
            99,
        )
    )

    if difference_limit == 0:
        difference_limit = 1.0

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17, 4),
    )

    panels = [
        (
            original_slice,
            "Original",
            "gray",
            vmin,
            vmax,
        ),
        (
            noisy_slice,
            f"Noisy {level}%",
            "gray",
            vmin,
            vmax,
        ),
        (
            denoised_slice,
            f"Real/Imag MP-PCA r={patch_radius}",
            "gray",
            vmin,
            vmax,
        ),
        (
            difference,
            "Denoised - Original",
            "RdBu_r",
            -difference_limit,
            difference_limit,
        ),
    ]

    for axis, panel in zip(
        axes,
        panels,
    ):
        (
            image,
            title,
            cmap,
            low,
            high,
        ) = panel

        axis.imshow(
            image.T,
            cmap=cmap,
            origin="lower",
            vmin=low,
            vmax=high,
        )

        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"Noise={level}% | "
        f"Patch radius={patch_radius} | "
        f"Slice={slice_index} | "
        f"Echo={echo_index + 1}"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading original: {ORIGINAL_PATH}",
        flush=True,
    )

    original_mat = loadmat(
        ORIGINAL_PATH,
        variable_names=[
            "meas_gre",
            "mask_brain",
            "te_gre",
        ],
    )

    original = original_mat[
        "meas_gre"
    ]

    mask = original_mat[
        "mask_brain"
    ].astype(bool)

    te_gre = original_mat.get(
        "te_gre"
    )

    if original.shape != (
        256,
        224,
        176,
        6,
    ):
        raise ValueError(
            f"Unexpected original shape: {original.shape}"
        )

    if mask.shape != original.shape[:3]:
        raise ValueError(
            f"Mask shape mismatch: {mask.shape}"
        )

    if not np.iscomplexobj(original):
        raise ValueError(
            "Original image is not complex"
        )

    if not (
        0
        <= args.slice_index
        < original.shape[2]
    ):
        raise ValueError(
            f"Invalid slice index: {args.slice_index}"
        )

    if not (
        0
        <= args.echo_index
        < original.shape[3]
    ):
        raise ValueError(
            f"Invalid echo index: {args.echo_index}"
        )

    print(
        f"Original shape: {original.shape}",
        flush=True,
    )

    print(
        f"Mask voxels: {int(mask.sum()):,}",
        flush=True,
    )

    for level in args.levels:
        noisy_path = (
            NOISY_DIR
            / f"noisy_{level}.mat"
        )

        if not noisy_path.exists():
            raise FileNotFoundError(
                noisy_path
            )

        print(
            f"\nLoading noisy data: {noisy_path}",
            flush=True,
        )

        noisy_mat = loadmat(
            noisy_path,
            variable_names=[
                "noisy_real",
                "noisy_imag",
                "noise_level_percent",
                "sigma",
                "seed",
            ],
        )

        noisy_real = noisy_mat[
            "noisy_real"
        ].astype(
            np.float32,
            copy=False,
        )

        noisy_imag = noisy_mat[
            "noisy_imag"
        ].astype(
            np.float32,
            copy=False,
        )

        if noisy_real.shape != original.shape:
            raise ValueError(
                "Noisy real shape mismatch: "
                f"{noisy_real.shape}"
            )

        if noisy_imag.shape != original.shape:
            raise ValueError(
                "Noisy imaginary shape mismatch: "
                f"{noisy_imag.shape}"
            )

        if not np.isfinite(
            noisy_real
        ).all():
            raise ValueError(
                f"Noise {level}% real contains NaN or Inf"
            )

        if not np.isfinite(
            noisy_imag
        ).all():
            raise ValueError(
                f"Noise {level}% imaginary contains NaN or Inf"
            )

        for patch_radius in args.patch_radii:
            output_path = (
                OUTPUT_DIR
                / (
                    "denoised_real_imag_"
                    f"{level}_r{patch_radius}.mat"
                )
            )

            preview_path = (
                OUTPUT_DIR
                / (
                    "denoised_real_imag_"
                    f"{level}_r{patch_radius}.png"
                )
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                print(
                    f"Skipping existing result: {output_path}",
                    flush=True,
                )
                continue

            print(
                "\n"
                f"Noise={level}% | "
                f"patch_radius={patch_radius}",
                flush=True,
            )

            print(
                "Running Real MP-PCA...",
                flush=True,
            )

            real_started = (
                time.perf_counter()
            )

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):
                denoised_real = mppca(
                    noisy_real,
                    mask=mask,
                    patch_radius=patch_radius,
                    out_dtype=np.float32,
                )

            real_elapsed = (
                time.perf_counter()
                - real_started
            )

            print(
                "Real MP-PCA finished: "
                f"{real_elapsed / 60:.2f} min",
                flush=True,
            )

            print(
                "Running Imaginary MP-PCA...",
                flush=True,
            )

            imag_started = (
                time.perf_counter()
            )

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):
                denoised_imag = mppca(
                    noisy_imag,
                    mask=mask,
                    patch_radius=patch_radius,
                    out_dtype=np.float32,
                )

            imag_elapsed = (
                time.perf_counter()
                - imag_started
            )

            print(
                "Imaginary MP-PCA finished: "
                f"{imag_elapsed / 60:.2f} min",
                flush=True,
            )

            validate_result(
                denoised_real,
                denoised_imag,
                mask,
            )

            payload = {
                "denoised_real":
                    denoised_real,

                "denoised_imag":
                    denoised_imag,

                "patch_radius":
                    np.array(
                        [[patch_radius]],
                        dtype=np.int16,
                    ),

                "noise_level_percent":
                    np.array(
                        [[level]],
                        dtype=np.int16,
                    ),

                "real_elapsed_seconds":
                    np.array(
                        [[real_elapsed]],
                        dtype=np.float64,
                    ),

                "imag_elapsed_seconds":
                    np.array(
                        [[imag_elapsed]],
                        dtype=np.float64,
                    ),
            }

            for key in (
                "sigma",
                "seed",
            ):
                if key in noisy_mat:
                    payload[key] = (
                        noisy_mat[key]
                    )

            if te_gre is not None:
                payload["te_gre"] = te_gre

            print(
                f"Saving: {output_path}",
                flush=True,
            )

            savemat(
                output_path,
                payload,
                do_compression=False,
            )

            save_preview(
                original=original,
                noisy_real=noisy_real,
                noisy_imag=noisy_imag,
                denoised_real=denoised_real,
                denoised_imag=denoised_imag,
                mask=mask,
                output_path=preview_path,
                level=level,
                patch_radius=patch_radius,
                slice_index=args.slice_index,
                echo_index=args.echo_index,
            )

            print(
                f"Saved: {preview_path}",
                flush=True,
            )

            del (
                denoised_real,
                denoised_imag,
                payload,
            )

            gc.collect()

        del (
            noisy_mat,
            noisy_real,
            noisy_imag,
        )

        gc.collect()

    print(
        "\nAll requested Real/Imaginary jobs completed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
