import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio


def _to_1d(arr):
    arr = np.asarray(arr)
    return np.squeeze(arr)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from a PGNN/HPD .mat results file")
    parser.add_argument("--results_file", required=True, type=str, help="Path to *_results.mat file")
    parser.add_argument("--out_file", default="./results/training_curves.png", type=str, help="Output image path")
    parser.add_argument("--title", default="Training Curves", type=str, help="Figure title")
    args = parser.parse_args()

    mat = spio.loadmat(args.results_file)

    train_rmse = _to_1d(mat["train_rmse"])
    val_rmse = _to_1d(mat["val_rmse"])
    test_rmse = float(_to_1d(mat["test_rmse"])[0]) if _to_1d(mat["test_rmse"]).shape else float(_to_1d(mat["test_rmse"]))

    epochs = np.arange(1, len(train_rmse) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_rmse, label="Train RMSE", linewidth=2)
    plt.plot(epochs, val_rmse, label="Validation RMSE", linewidth=2)
    plt.axhline(test_rmse, linestyle="--", linewidth=1.8, label="Test RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out_file, dpi=160)

    print("Saved plot:", args.out_file)
    print("Last train RMSE:", float(train_rmse[-1]))
    print("Last val RMSE:", float(val_rmse[-1]))
    print("Test RMSE:", test_rmse)


if __name__ == "__main__":
    main()
