from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from gmm.gibbs_sampler import BayesianGMMGibbsSampler


FEATURE_COLUMNS = [
    "field_size",
    "distance",
    "favorite_odds",
    "second_favorite_odds",
    "third_favorite_odds",
    "odds_gap",
    "top3_odds_mean",
    "odds_entropy",
]


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # 定数列対策
    std[std == 0] = 1.0

    X_std = (X - mean) / std
    return X_std, mean, std


def make_initial_z_by_market_uncertainty(
    df: pd.DataFrame,
    n_components: int,
) -> np.ndarray:
    """
    オッズ分布のエントロピーを使って初期クラスタを作る。
    あくまで初期値であり、最終的なクラスタはGibbsサンプリングで更新される。
    """

    values = df["odds_entropy"].to_numpy()

    ranks = pd.qcut(
        values,
        q=n_components,
        labels=False,
        duplicates="drop",
    )

    z = np.asarray(ranks, dtype=int)

    if z.max() + 1 < n_components:
        rng = np.random.default_rng(0)
        z = rng.integers(0, n_components, size=len(df))

    return z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/race_features.csv")
    parser.add_argument("--output", type=str, default="results/samples/gibbs_samples.npz")
    parser.add_argument("--labeled-output", type=str, default="results/tables/race_features_with_cluster.csv")
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    labeled_output_path = Path(args.labeled_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading processed race features...")
    df = pd.read_csv(input_path)

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_std, x_mean, x_std = standardize(X)

    initial_z = make_initial_z_by_market_uncertainty(df, args.n_components)

    print(f"Number of races: {len(df)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of clusters: {args.n_components}")

    sampler = BayesianGMMGibbsSampler(
        n_components=args.n_components,
        alpha=1.0,
        kappa0=0.01,
        psi_scale=1.0,
        random_state=args.seed,
    )

    result = sampler.fit(
        X_std,
        n_iter=args.n_iter,
        burn_in=args.burn_in,
        thin=args.thin,
        initial_z=initial_z,
        verbose=True,
    )

    df["cluster"] = result.z

    for k in range(args.n_components):
        df[f"cluster_prob_{k}"] = result.cluster_prob[:, k]

    np.savez(
        output_path,
        feature_columns=np.array(FEATURE_COLUMNS),
        x_mean=x_mean,
        x_std=x_std,
        z=result.z,
        cluster_prob=result.cluster_prob,
        pi_samples=result.pi_samples,
        mu_samples=result.mu_samples,
        final_pi=result.final_pi,
        final_mu=result.final_mu,
        final_sigma=result.final_sigma,
    )

    df.to_csv(labeled_output_path, index=False, encoding="utf-8-sig")

    print(f"Saved samples: {output_path}")
    print(f"Saved labeled data: {labeled_output_path}")

    print("Final cluster counts:")
    print(df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()