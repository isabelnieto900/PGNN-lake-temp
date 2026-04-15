import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as spio
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def density(temp: np.ndarray) -> np.ndarray:
    temp = np.asarray(temp, dtype=float)
    return 1000 * (
        1
        - (temp + 288.9414) * (temp - 3.9863) ** 2 / (508929.2 * (temp + 68.12963))
    )


def evaluate_physical_inconsistency(model, ux1: np.ndarray, ux2: np.ndarray) -> Tuple[float, float]:
    tolerance = 0.0
    pred1 = model.predict(ux1).reshape(-1, 1)
    pred2 = model.predict(ux2).reshape(-1, 1)
    udendiff = density(pred1) - density(pred2)
    percentage_phy_incon = float(np.mean(udendiff > tolerance))
    phy_loss = float(np.mean(np.maximum(udendiff, 0.0)))
    return phy_loss, percentage_phy_incon


def load_lake_data(data_dir: Path, lake_name: str, tr_size: int, use_yphy: int):
    sup = spio.loadmat(
        data_dir / f"{lake_name}.mat",
        squeeze_me=True,
        variable_names=["Y", "Xc_doy"],
    )
    unsup = spio.loadmat(
        data_dir / f"{lake_name}_sampled.mat",
        squeeze_me=True,
        variable_names=["Xc_doy1", "Xc_doy2"],
    )

    x = np.asarray(sup["Xc_doy"], dtype=float)
    y = np.asarray(sup["Y"], dtype=float).reshape(-1)
    ux1 = np.asarray(unsup["Xc_doy1"], dtype=float)
    ux2 = np.asarray(unsup["Xc_doy2"], dtype=float)

    if use_yphy == 0:
        x = x[:, :-1]
        ux1 = ux1[:, :-1]
        ux2 = ux2[:, :-1]

    train_x, train_y = x[:tr_size, :], y[:tr_size]
    test_x, test_y = x[tr_size:, :], y[tr_size:]
    return train_x, train_y, test_x, test_y, ux1, ux2


def run_svm_for_lake(args, lake_name: str):
    train_x, train_y, test_x, test_y, ux1, ux2 = load_lake_data(
        args.data_dir, lake_name, args.tr_size, args.use_yphy
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=args.c,
                    epsilon=args.epsilon,
                    gamma=args.gamma,
                ),
            ),
        ]
    )

    model.fit(train_x, train_y)

    test_pred = model.predict(test_x)
    test_rmse = float(np.sqrt(mean_squared_error(test_y, test_pred)))
    # SVM-RBF prediction on the full unsupervised set can be very slow; use a reproducible subset.
    if args.max_unsup_samples > 0 and ux1.shape[0] > args.max_unsup_samples:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(ux1.shape[0], size=args.max_unsup_samples, replace=False)
        ux1_eval = ux1[idx]
        ux2_eval = ux2[idx]
    else:
        ux1_eval = ux1
        ux2_eval = ux2

    phy_consistency, percentage_phy_incon = evaluate_physical_inconsistency(model, ux1_eval, ux2_eval)

    exp_name = (
        f"svm_{lake_name}_trsize{args.tr_size}_usePhy{args.use_yphy}_"
        f"C{args.c}_eps{args.epsilon}_gamma{args.gamma}"
    ).replace(".", "pt")

    save_dir = args.save_dir / "SVM" / lake_name
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / f"{exp_name}_results.mat"

    spio.savemat(
        results_path,
        {
            "test_rmse": test_rmse,
            "rmse": test_rmse,
            "phy_consistency": phy_consistency,
            "physical_inconsistency": percentage_phy_incon,
            "percentage_phy_incon": percentage_phy_incon,
            "tr_size": args.tr_size,
            "use_YPhy": args.use_yphy,
            "C": args.c,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
            "max_unsup_samples": args.max_unsup_samples,
            "seed": args.seed,
        },
    )

    print(
        "lake:",
        lake_name,
        "test_rmse:",
        test_rmse,
        "physical_inconsistency:",
        percentage_phy_incon,
        "saved:",
        results_path,
    )


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="SVM-RBF baseline for lake temperature")
    parser.add_argument(
        "--dataset",
        choices=["mendota", "mille_lacs", "all"],
        default="all",
        help="Lake dataset to run",
    )
    parser.add_argument("--data_dir", type=Path, default=repo_root / "datasets")
    parser.add_argument("--save_dir", type=Path, default=repo_root / "results")
    parser.add_argument("--tr_size", type=int, default=3000)
    parser.add_argument(
        "--use_yphy",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use physics simulation feature as input (0 for pure black-box baseline)",
    )
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--max_unsup_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lakes = ["mendota", "mille_lacs"] if args.dataset == "all" else [args.dataset]
    for lake in lakes:
        run_svm_for_lake(args, lake)