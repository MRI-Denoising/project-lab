"""Run Magnitude MP-PCA for 10%, 30%, and 50% noisy MRI data."""

from pathlib import Path
import gc
import time

import matplotlib

# 서버에는 화면이 없으므로 파일 저장용 백엔드를 사용합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat, savemat
from dipy.denoise.localpca import mppca


# ---------------------------------------------------------
# 1. 프로젝트 경로와 실험 조건
# ---------------------------------------------------------

# 현재 파일 위치:
# project-lab/03_denoise_magnitude.py
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
    / "magnitude"
)

NOISE_LEVELS = [10, 30, 50]
PATCH_RADIUS = 2

SLICE_INDEX = 88
ECHO_INDEX = 0

SAVE_COMPRESSED = False

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# 2. 입력 파일 확인
# ---------------------------------------------------------

if not ORIGINAL_PATH.exists():
    raise FileNotFoundError(
        f"원본 파일이 없습니다: {ORIGINAL_PATH}"
    )

for level in NOISE_LEVELS:
    noisy_path = (
        NOISY_DIR
        / f"noisy_{level:02d}.mat"
    )

    if not noisy_path.exists():
        raise FileNotFoundError(
            f"Noisy 파일이 없습니다: {noisy_path}"
        )

print("Project root:", PROJECT_ROOT, flush=True)
print("Original:", ORIGINAL_PATH, flush=True)
print("Noisy directory:", NOISY_DIR, flush=True)
print("Output directory:", OUTPUT_DIR, flush=True)


# ---------------------------------------------------------
# 3. 원본 Magnitude와 마스크 불러오기
# ---------------------------------------------------------

print("\n원본 데이터를 불러오는 중...", flush=True)

original_mat = loadmat(
    ORIGINAL_PATH,
    variable_names=[
        "meas_gre",
        "mask_brain",
        "te_gre",
    ],
)

original_complex = original_mat["meas_gre"]
mask = original_mat["mask_brain"].astype(bool)
te_gre = original_mat.get("te_gre")

if original_complex.ndim != 4:
    raise ValueError(
        f"원본이 4D 데이터가 아닙니다: "
        f"{original_complex.shape}"
    )

if not np.iscomplexobj(original_complex):
    raise ValueError(
        f"원본이 complex 데이터가 아닙니다: "
        f"{original_complex.dtype}"
    )

if mask.shape != original_complex.shape[:3]:
    raise ValueError(
        f"Mask shape이 일치하지 않습니다: "
        f"{mask.shape}, {original_complex.shape}"
    )

original_magnitude = np.abs(
    original_complex
).astype(np.float32)

del original_complex
del original_mat

gc.collect()

print(
    "Original magnitude:",
    original_magnitude.shape,
    original_magnitude.dtype,
    flush=True,
)


# ---------------------------------------------------------
# 4. 10%, 30%, 50% 순서대로 처리
# ---------------------------------------------------------

for level in NOISE_LEVELS:
    print("\n" + "=" * 60, flush=True)
    print(
        f"Noise {level}% Magnitude MP-PCA 시작",
        flush=True,
    )
    print("=" * 60, flush=True)

    noisy_path = (
        NOISY_DIR
        / f"noisy_{level:02d}.mat"
    )

    output_path = (
        OUTPUT_DIR
        / f"denoised_magnitude_{level:02d}.mat"
    )

    preview_path = (
        OUTPUT_DIR
        / f"denoised_magnitude_{level:02d}.png"
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

    noisy_real = noisy_mat.pop(
        "noisy_real"
    ).astype(
        np.float32,
        copy=False,
    )

    noisy_imag = noisy_mat.pop(
        "noisy_imag"
    ).astype(
        np.float32,
        copy=False,
    )

    if noisy_real.shape != original_magnitude.shape:
        raise ValueError(
            f"Noisy Real shape 오류: "
            f"{noisy_real.shape}"
        )

    if noisy_imag.shape != original_magnitude.shape:
        raise ValueError(
            f"Noisy Imaginary shape 오류: "
            f"{noisy_imag.shape}"
        )

    # Magnitude 6채널 생성
    noisy_magnitude = np.hypot(
        noisy_real,
        noisy_imag,
    ).astype(
        np.float32,
        copy=False,
    )

    del noisy_real
    del noisy_imag

    gc.collect()

    print(
        "MP-PCA input:",
        noisy_magnitude.shape,
        noisy_magnitude.dtype,
        flush=True,
    )
    print(
        f"patch_radius={PATCH_RADIUS}",
        flush=True,
    )

    started = time.perf_counter()

    denoised_magnitude = mppca(
        noisy_magnitude,
        mask=mask,
        patch_radius=PATCH_RADIUS,
        out_dtype=np.float32,
    )

    elapsed_seconds = (
        time.perf_counter() - started
    )

    elapsed_minutes = (
        elapsed_seconds / 60
    )

    print(
        f"MP-PCA 완료: {elapsed_minutes:.2f}분",
        flush=True,
    )


    # -----------------------------------------------------
    # 5. MAT 결과 저장
    # -----------------------------------------------------

    result = {
        "denoised_magnitude": (
            denoised_magnitude
        ),
        "patch_radius": np.array(
            [[PATCH_RADIUS]],
            dtype=np.int16,
        ),
        "elapsed_seconds": np.array(
            [[elapsed_seconds]],
            dtype=np.float64,
        ),
    }

    for key in [
        "noise_level_percent",
        "sigma",
        "seed",
    ]:
        if key in noisy_mat:
            result[key] = noisy_mat[key]

    if te_gre is not None:
        result["te_gre"] = te_gre

    print(
        "MAT 저장 중:",
        output_path,
        flush=True,
    )

    savemat(
        output_path,
        result,
        do_compression=SAVE_COMPRESSED,
    )


    # -----------------------------------------------------
    # 6. 결과 시각화
    # -----------------------------------------------------

    original2d = original_magnitude[
        :, :, SLICE_INDEX, ECHO_INDEX
    ]

    noisy2d = noisy_magnitude[
        :, :, SLICE_INDEX, ECHO_INDEX
    ]

    denoised2d = denoised_magnitude[
        :, :, SLICE_INDEX, ECHO_INDEX
    ]

    mask2d = mask[:, :, SLICE_INDEX]

    vmin, vmax = np.percentile(
        original2d[mask2d],
        [1, 99],
    )

    difference = np.where(
        mask2d,
        denoised2d - original2d,
        0,
    )

    diff_limit = max(
        float(
            np.percentile(
                np.abs(
                    difference[mask2d]
                ),
                99,
            )
        ),
        1e-12,
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 4),
    )

    panels = [
        (
            original2d,
            "Original magnitude",
            "gray",
            vmin,
            vmax,
        ),
        (
            noisy2d,
            f"Noisy magnitude ({level}%)",
            "gray",
            vmin,
            vmax,
        ),
        (
            denoised2d,
            "Magnitude MP-PCA",
            "gray",
            vmin,
            vmax,
        ),
        (
            difference,
            "Denoised - Original",
            "RdBu_r",
            -diff_limit,
            diff_limit,
        ),
    ]

    for ax, (
        data,
        title,
        cmap,
        low,
        high,
    ) in zip(axes, panels):
        ax.imshow(
            data.T,
            cmap=cmap,
            origin="lower",
            vmin=low,
            vmax=high,
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Noise {level}% | "
        f"patch radius={PATCH_RADIUS} | "
        f"{elapsed_minutes:.2f} min"
    )

    plt.tight_layout()

    plt.savefig(
        preview_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "PNG 저장:",
        preview_path,
        flush=True,
    )


    # -----------------------------------------------------
    # 7. 다음 noise level을 위해 메모리 정리
    # -----------------------------------------------------

    del noisy_mat
    del noisy_magnitude
    del denoised_magnitude
    del result
    del original2d
    del noisy2d
    del denoised2d
    del difference

    gc.collect()

    print(
        f"Noise {level}% 처리 완료",
        flush=True,
    )


print(
    "\n모든 Magnitude MP-PCA 작업 완료",
    flush=True,
)
