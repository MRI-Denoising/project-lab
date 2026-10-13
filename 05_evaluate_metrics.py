"""Evaluate MRI denoising results using error SNR, RMSE, and masked SSIM."""

from __future__ import annotations

import gc
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity


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

MAGNITUDE_DIR = (
    PROJECT_ROOT
    / "results"
    / "magnitude"
)

REAL_IMAG_DIR = (
    PROJECT_ROOT
    / "results"
    / "real_imag"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

NOISE_LEVELS = [10, 30, 50]
PATCH_RADII = [2, 3]


def masked_ssim_2d(
    reference,
    test,
    mask_2d,
    data_range,
):
    """Calculate SSIM and average only inside an eroded brain mask."""

    valid_mask = binary_erosion(
        mask_2d,
        iterations=5,
    )

    if not valid_mask.any():
        valid_mask = mask_2d

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            RuntimeWarning,
        )

        _, score_map = structural_similarity(
            reference,
            test,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            channel_axis=None,
            full=True,
        )

    values = score_map[valid_mask]
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise RuntimeError(
            "No finite SSIM values inside mask"
        )

    return float(values.mean())


def evaluate_volume(
    original_magnitude,
    test_magnitude,
    mask,
    noise_level,
    method,
    patch_radius,
):
    """Calculate echo-wise error SNR, RMSE, and masked SSIM."""

    if test_magnitude.shape != original_magnitude.shape:
        raise ValueError(
            f"{method}: shape mismatch: "
            f"{test_magnitude.shape} != "
            f"{original_magnitude.shape}"
        )

    if not np.isfinite(
        test_magnitude[mask]
    ).all():
        raise ValueError(
            f"{method}: NaN or Inf inside mask"
        )

    rows = []

    for echo in range(
        original_magnitude.shape[-1]
    ):
        reference_echo = (
            original_magnitude[..., echo]
        )

        test_echo = (
            test_magnitude[..., echo]
        )

        reference_values = (
            reference_echo[mask]
            .astype(np.float64)
        )

        test_values = (
            test_echo[mask]
            .astype(np.float64)
        )

        error = (
            test_values
            - reference_values
        )

        signal_rms = float(
            np.sqrt(
                np.mean(
                    reference_values ** 2
                )
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    error ** 2
                )
            )
        )

        if rmse == 0:
            snr_linear = float("inf")
            snr_db = float("inf")
        else:
            snr_linear = (
                signal_rms
                / rmse
            )

            snr_db = float(
                20.0
                * np.log10(
                    snr_linear
                )
            )

        # 같은 echo의 모든 방법에서 동일한 범위를 사용
        data_range = float(
            reference_values.max()
            - reference_values.min()
        )

        if data_range <= 0:
            raise ValueError(
                f"Echo {echo + 1}: "
                "invalid original data range"
            )

        slice_scores = []

        for slice_index in range(
            original_magnitude.shape[2]
        ):
            mask_2d = mask[
                :,
                :,
                slice_index,
            ]

            if not mask_2d.any():
                continue

            score = masked_ssim_2d(
                reference_echo[
                    :,
                    :,
                    slice_index,
                ],
                test_echo[
                    :,
                    :,
                    slice_index,
                ],
                mask_2d,
                data_range,
            )

            slice_scores.append(
                score
            )

        mean_ssim = float(
            np.mean(
                slice_scores
            )
        )

        row = {
            "Noise_percent":
                noise_level,

            "Method":
                method,

            "Patch_radius":
                patch_radius,

            "Echo":
                echo + 1,

            "SNR_linear":
                snr_linear,

            "SNR_dB":
                snr_db,

            "RMSE":
                rmse,

            "SSIM":
                mean_ssim,
        }

        rows.append(row)

        print(
            f"Noise={noise_level:2d}% | "
            f"{method:<16} | "
            f"r={str(patch_radius):>4} | "
            f"echo={echo + 1} | "
            f"SNR={snr_db:8.3f} dB | "
            f"RMSE={rmse:10.6f} | "
            f"SSIM={mean_ssim:.6f}",
            flush=True,
        )

    return rows


def load_noisy_magnitude(
    noise_level,
):
    path = (
        NOISY_DIR
        / f"noisy_{noise_level}.mat"
    )

    mat = loadmat(
        path,
        variable_names=[
            "noisy_real",
            "noisy_imag",
        ],
    )

    magnitude = np.hypot(
        mat["noisy_real"],
        mat["noisy_imag"],
    ).astype(
        np.float32,
        copy=False,
    )

    return magnitude


def load_magnitude_result(
    noise_level,
):
    path = (
        MAGNITUDE_DIR
        / (
            "denoised_magnitude_"
            f"{noise_level}.mat"
        )
    )

    mat = loadmat(
        path,
        variable_names=[
            "denoised_magnitude",
        ],
    )

    return mat[
        "denoised_magnitude"
    ].astype(
        np.float32,
        copy=False,
    )


def load_real_imag_result(
    noise_level,
    patch_radius,
):
    path = (
        REAL_IMAG_DIR
        / (
            "denoised_real_imag_"
            f"{noise_level}_"
            f"r{patch_radius}.mat"
        )
    )

    mat = loadmat(
        path,
        variable_names=[
            "denoised_real",
            "denoised_imag",
        ],
    )

    magnitude = np.hypot(
        mat["denoised_real"],
        mat["denoised_imag"],
    ).astype(
        np.float32,
        copy=False,
    )

    return magnitude


def add_improvement_columns(
    frame,
):
    baseline = (
        frame[
            frame["Method"] == "Noisy"
        ][
            [
                "Noise_percent",
                "Echo",
                "SNR_dB",
                "RMSE",
                "SSIM",
            ]
        ]
        .rename(
            columns={
                "SNR_dB":
                    "Noisy_SNR_dB",

                "RMSE":
                    "Noisy_RMSE",

                "SSIM":
                    "Noisy_SSIM",
            }
        )
    )

    result = frame.merge(
        baseline,
        on=[
            "Noise_percent",
            "Echo",
        ],
        how="left",
    )

    result["SNR_gain_dB"] = (
        result["SNR_dB"]
        - result["Noisy_SNR_dB"]
    )

    result["RMSE_reduction_percent"] = (
        (
            result["Noisy_RMSE"]
            - result["RMSE"]
        )
        / result["Noisy_RMSE"]
        * 100.0
    )

    result["SSIM_gain"] = (
        result["SSIM"]
        - result["Noisy_SSIM"]
    )

    return result


def save_metric_plot(
    frame,
    metric,
    ylabel,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        sharey=False,
    )

    for axis, noise_level in zip(
        axes,
        NOISE_LEVELS,
    ):
        subset = frame[
            frame["Noise_percent"]
            == noise_level
        ].copy()

        for (
            method,
            patch_radius,
        ), group in subset.groupby(
            [
                "Method",
                "Patch_radius",
            ],
            dropna=False,
        ):
            if pd.isna(
                patch_radius
            ):
                label = method
            else:
                label = (
                    f"{method} "
                    f"r{int(patch_radius)}"
                )

            group = group.sort_values(
                "Echo"
            )

            axis.plot(
                group["Echo"],
                group[metric],
                marker="o",
                label=label,
            )

        axis.set_title(
            f"Noise {noise_level}%"
        )

        axis.set_xlabel(
            "Echo"
        )

        axis.set_xticks(
            range(1, 7)
        )

        axis.grid(
            alpha=0.3
        )

    axes[0].set_ylabel(
        ylabel
    )

    axes[-1].legend(
        bbox_to_anchor=(
            1.04,
            1.0,
        ),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
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
        ],
    )

    original_magnitude = np.abs(
        original_mat["meas_gre"]
    ).astype(
        np.float32
    )

    mask = original_mat[
        "mask_brain"
    ].astype(bool)

    del original_mat
    gc.collect()

    if original_magnitude.shape != (
        256,
        224,
        176,
        6,
    ):
        raise ValueError(
            "Unexpected original shape: "
            f"{original_magnitude.shape}"
        )

    rows = []

    for noise_level in NOISE_LEVELS:
        print(
            "\n"
            f"Evaluating noise "
            f"{noise_level}%",
            flush=True,
        )

        noisy_magnitude = (
            load_noisy_magnitude(
                noise_level
            )
        )

        rows.extend(
            evaluate_volume(
                original_magnitude,
                noisy_magnitude,
                mask,
                noise_level,
                "Noisy",
                None,
            )
        )

        del noisy_magnitude
        gc.collect()

        magnitude_result = (
            load_magnitude_result(
                noise_level
            )
        )

        rows.extend(
            evaluate_volume(
                original_magnitude,
                magnitude_result,
                mask,
                noise_level,
                "Magnitude",
                2,
            )
        )

        del magnitude_result
        gc.collect()

        for patch_radius in PATCH_RADII:
            real_imag_result = (
                load_real_imag_result(
                    noise_level,
                    patch_radius,
                )
            )

            rows.extend(
                evaluate_volume(
                    original_magnitude,
                    real_imag_result,
                    mask,
                    noise_level,
                    "Real-Imag",
                    patch_radius,
                )
            )

            del real_imag_result
            gc.collect()

    detail = pd.DataFrame(
        rows
    )

    detail = add_improvement_columns(
        detail
    )

    detail = detail.sort_values(
        [
            "Noise_percent",
            "Method",
            "Patch_radius",
            "Echo",
        ],
        na_position="first",
    ).reset_index(
        drop=True
    )

    summary = (
        detail
        .groupby(
            [
                "Noise_percent",
                "Method",
                "Patch_radius",
            ],
            dropna=False,
        )
        [
            [
                "SNR_dB",
                "SNR_gain_dB",
                "RMSE",
                "RMSE_reduction_percent",
                "SSIM",
                "SSIM_gain",
            ]
        ]
        .mean()
        .reset_index()
    )

    detail_csv = (
        OUTPUT_DIR
        / "metrics_by_echo.csv"
    )

    summary_csv = (
        OUTPUT_DIR
        / "metrics_summary.csv"
    )

    excel_path = (
        OUTPUT_DIR
        / "metrics_results.xlsx"
    )

    detail.to_csv(
        detail_csv,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        detail.to_excel(
            writer,
            sheet_name="EchoDetail",
            index=False,
        )

        summary.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

    save_metric_plot(
        detail,
        metric="SNR_dB",
        ylabel="Error SNR (dB)",
        output_path=(
            OUTPUT_DIR
            / "snr_by_echo.png"
        ),
    )

    save_metric_plot(
        detail,
        metric="SSIM",
        ylabel="Masked SSIM",
        output_path=(
            OUTPUT_DIR
            / "ssim_by_echo.png"
        ),
    )

    print(
        "\nSummary:",
        flush=True,
    )

    print(
        summary.round(4).to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nSaved:",
        flush=True,
    )

    for path in (
        detail_csv,
        summary_csv,
        excel_path,
        OUTPUT_DIR
        / "snr_by_echo.png",
        OUTPUT_DIR
        / "ssim_by_echo.png",
    ):
        print(
            path,
            flush=True,
        )


if __name__ == "__main__":
    main()
