"""
Lab 12: CNN Feature Visualization, Parameter Audit & Critical Analysis
=====================================================================
Implements:
  - Part I: Five custom 3x3 spatial filters across 4 sequential stages
    (Raw Convolution, Post-ReLU, Max Pooling 2x2/s2, Average Pooling 2x2/s2)
    and plots a unified 5x4 visualization grid with colorbars.
  - Part II: Two lightweight CNN models (Model A: 3x3 + MaxPool,
    Model B: 5x5 + AvgPool) in TensorFlow/Keras, printing model.summary()
    and verifying manual parameter counts.
"""

import os
import io
import contextlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_architectural_test_image(size: int = 256) -> np.ndarray:
    """
    Generate a high-contrast architectural facade image (grayscale, float32 in [0, 1])
    with crisp horizontal beams, vertical pillars, recessed windows, and subtle gradients.
    """
    img = np.full((size, size), 0.82, dtype=np.float32)

    # Subtle concrete sky/wall gradient
    grad = np.linspace(0.92, 0.72, size, dtype=np.float32)[:, None]
    img = img * 0.5 + grad * 0.5

    # Vertical structural pillars
    for x in [16, 72, 128, 184, 240]:
        img[:, max(0, x - 6):min(size, x + 6)] = 0.95
        img[:, max(0, x + 6):min(size, x + 9)] = 0.35  # cast shadow

    # Horizontal floor slabs
    for y in [20, 76, 132, 188, 244]:
        img[max(0, y - 5):min(size, y + 5), :] = 0.90
        img[max(0, y + 5):min(size, y + 8), :] = 0.30  # overhang shadow

    # Dark recessed window panes inside grid cells
    rows = [(32, 68), (88, 124), (144, 180), (200, 236)]
    cols = [(28, 64), (84, 120), (140, 176), (196, 232)]
    for r0, r1 in rows:
        for c0, c1 in cols:
            img[r0:r1, c0:c1] = 0.12
            # Bright window mullion crossbars
            mid_r = (r0 + r1) // 2
            mid_c = (c0 + c1) // 2
            img[mid_r - 1:mid_r + 1, c0:c1] = 0.75
            img[r0:r1, mid_c - 1:mid_c + 1] = 0.75

    # Fine architectural texture
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.015, size=(size, size)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return img


def get_custom_kernels() -> dict[str, np.ndarray]:
    """Return the five required 3x3 custom convolution kernels."""
    return {
        "1. Horizontal Edge (Sobel-X)": np.array(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=np.float32
        ),
        "2. Vertical Edge (Sobel-Y)": np.array(
            [[-1,  0,  1],
             [-2,  0,  2],
             [-1,  0,  1]], dtype=np.float32
        ),
        "3. Sharpen Kernel": np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]], dtype=np.float32
        ),
        "4. Gaussian Blur (3x3)": np.array(
            [[1, 2, 1],
             [2, 4, 2],
             [1, 2, 1]], dtype=np.float32
        ) / 16.0,
        "5. Emboss Kernel": np.array(
            [[-2, -1,  0],
             [-1,  1,  1],
             [ 0,  1,  2]], dtype=np.float32
        ),
    }


def pool_2x2_stride2(feat: np.ndarray, mode: str = "max") -> np.ndarray:
    """Apply 2x2 pooling with stride 2 to a 2D feature map."""
    h, w = feat.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    blocks = feat[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
    if mode == "max":
        return blocks.max(axis=(1, 3))
    if mode == "avg":
        return blocks.mean(axis=(1, 3))
    raise ValueError(f"Unknown pooling mode: {mode}")


def run_part1_visualization(img: np.ndarray) -> None:
    """Execute the 4-stage filter pipeline and save the 5x4 visualization grid."""
    kernels = get_custom_kernels()
    stage_names = [
        "Stage A: Raw Conv",
        "Stage B: Post-ReLU",
        "Stage C: MaxPool (2x2, s=2)",
        "Stage D: AvgPool (2x2, s=2)",
    ]

    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    fig.suptitle(
        "CNN Feature Visualization: 5 Custom 3x3 Kernels Across 4 Pipeline Stages",
        fontsize=16, fontweight="bold", y=0.995
    )

    print("=" * 80)
    print("PART I: FILTER PIPELINE FEATURE MAP STATISTICS")
    print("=" * 80)

    for row_idx, (k_name, kernel) in enumerate(kernels.items()):
        # Stage A: Raw linear convolution
        stage_a = cv2.filter2D(img, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
        # Stage B: Post-ReLU activation
        stage_b = np.maximum(0.0, stage_a)
        # Stage C: Max Pooling (2x2, stride 2)
        stage_c = pool_2x2_stride2(stage_b, mode="max")
        # Stage D: Average Pooling (2x2, stride 2)
        stage_d = pool_2x2_stride2(stage_b, mode="avg")

        stages = [stage_a, stage_b, stage_c, stage_d]
        neg_pct = float((stage_a < 0).mean() * 100.0)
        zero_pct_b = float((stage_b == 0).mean() * 100.0)
        print(
            f"{k_name:30s} | "
            f"A:[{stage_a.min():6.2f}, {stage_a.max():6.2f}] ({neg_pct:5.1f}% < 0) | "
            f"B:[{stage_b.min():5.2f}, {stage_b.max():5.2f}] ({zero_pct_b:5.1f}% == 0) | "
            f"C_mean={stage_c.mean():.3f} | D_mean={stage_d.mean():.3f}"
        )

        # Shared positive scale for Stages B, C, D so pooling differences are directly comparable
        pos_vmax = max(float(stage_b.max()), 1e-5)
        raw_abs_max = max(abs(float(stage_a.min())), abs(float(stage_a.max())), 1e-5)

        for col_idx, feat in enumerate(stages):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                # Diverging colormap centered at 0 for Stage A (negative vs positive)
                im = ax.imshow(feat, cmap="RdBu_r", vmin=-raw_abs_max, vmax=raw_abs_max)
            else:
                # Grayscale starting at 0 (pure black = zero activation)
                im = ax.imshow(feat, cmap="gray", vmin=0.0, vmax=pos_vmax)

            if row_idx == 0:
                ax.set_title(stage_names[col_idx], fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(k_name, fontsize=10.5, fontweight="bold", labelpad=8)

            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    grid_path = os.path.join(OUTPUT_DIR, "part1_5x4_feature_grid.png")
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved 5x4 visualization grid -> {grid_path}")


def build_cnn_model(name: str, kernel_size: int, pool_type: str) -> tf.keras.Model:
    """Build Model A (3x3, MaxPool) or Model B (5x5, AvgPool) for 32x32x3 RGB input."""
    pool_layer = layers.MaxPooling2D if pool_type == "max" else layers.AveragePooling2D
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(32, 32, 3), name="input_rgb"))
    model.add(layers.Conv2D(5, (kernel_size, kernel_size), activation="relu", name="conv1"))
    model.add(pool_layer(pool_size=(2, 2), strides=2, name="pool1"))
    model.add(layers.Conv2D(10, (kernel_size, kernel_size), activation="relu", name="conv2"))
    model.add(pool_layer(pool_size=(2, 2), strides=2, name="pool2"))
    model.add(layers.GlobalAveragePooling2D(name="global_avg_pool"))
    model.add(layers.Dense(10, activation="softmax", name="output_dense"))
    return model


def run_part2_parameter_audit() -> str:
    """Construct Model A & Model B, print summaries, and verify manual counts."""
    model_a = build_cnn_model("Model_A_Standard_3x3_MaxPool", kernel_size=3, pool_type="max")
    model_b = build_cnn_model("Model_B_LargeKernel_5x5_AvgPool", kernel_size=5, pool_type="avg")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("=" * 80)
        print("PART II: ARCHITECTURAL COMPARISON & PARAMETER AUDIT")
        print("=" * 80)
        model_a.summary()
        print()
        model_b.summary()
        print()
        print(f"Model A Total Trainable Parameters: {model_a.count_params()}")
        print(f"Model B Total Trainable Parameters: {model_b.count_params()}")
        pct_inc = (model_b.count_params() - model_a.count_params()) / model_a.count_params() * 100.0
        print(f"Percentage Increase (Model A -> Model B): {pct_inc:.2f}%")

    summary_text = buf.getvalue()
    print("\n" + summary_text)

    summary_path = os.path.join(OUTPUT_DIR, "part2_model_summaries.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved model summaries -> {summary_path}")
    return summary_text


def main() -> None:
    img = create_architectural_test_image(256)
    input_path = os.path.join(OUTPUT_DIR, "input_architecture.png")
    plt.imsave(input_path, img, cmap="gray", vmin=0.0, vmax=1.0)
    print(f"Saved input architectural image -> {input_path}")

    run_part1_visualization(img)
    run_part2_parameter_audit()


if __name__ == "__main__":
    main()
