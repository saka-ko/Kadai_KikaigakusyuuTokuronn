from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pi_trace(pi_samples: np.ndarray, out_path: Path) -> None:
    """混合比率 pi のサンプル列をプロットする。"""

    plt.figure(figsize=(8, 5))

    n_components = pi_samples.shape[1]
    for k in range(n_components):
        plt.plot(pi_samples[:, k], label=f"Cluster {k}")

    plt.xlabel("Saved sample index")
    plt.ylabel("Mixture weight")
    plt.title("Trace plot of mixture weights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cluster_probability_hist(df: pd.DataFrame, out_path: Path) -> None:
    """各データ点の最大クラスタ所属確率の分布をプロットする。"""

    prob_cols = [col for col in df.columns if col.startswith("cluster_prob_")]

    if not prob_cols:
        print("No cluster probability columns found.")
        return

    max_prob = df[prob_cols].max(axis=1)

    plt.figure(figsize=(8, 5))
    plt.hist(max_prob, bins=30)
    plt.xlabel("Maximum posterior cluster probability")
    plt.ylabel("Number of races")
    plt.title("Posterior certainty of cluster assignments")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Mean max posterior probability: {max_prob.mean():.4f}")
    print(f"Median max posterior probability: {max_prob.median():.4f}")


def plot_pi_posterior_boxplot(pi_samples: np.ndarray, out_path: Path) -> None:
    """混合比率 pi の事後サンプル分布を箱ひげ図で表示する。"""

    n_components = pi_samples.shape[1]

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [pi_samples[:, k] for k in range(n_components)],
        tick_labels=[f"Cluster {k}" for k in range(n_components)],
        showfliers=False,
    )
    plt.xlabel("Cluster")
    plt.ylabel("Mixture weight")
    plt.title("Posterior distribution of mixture weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=str,
        default="results/samples/gibbs_samples.npz",
    )
    parser.add_argument(
        "--clustered-data",
        type=str,
        default="results/tables/race_features_with_cluster.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="results/figures",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    clustered_data_path = Path(args.clustered_data)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Gibbs samples...")
    samples = np.load(samples_path, allow_pickle=True)
    pi_samples = samples["pi_samples"]

    print(f"Number of saved samples: {len(pi_samples)}")
    print("Posterior mean of pi:")
    print(np.round(pi_samples.mean(axis=0), 4))

    plot_pi_trace(
        pi_samples,
        figure_dir / "pi_trace_plot.png",
    )

    plot_pi_posterior_boxplot(
        pi_samples,
        figure_dir / "pi_posterior_boxplot.png",
    )

    print("Loading clustered data...")
    df = pd.read_csv(clustered_data_path)

    plot_cluster_probability_hist(
        df,
        figure_dir / "cluster_probability_hist.png",
    )

    print(f"Saved diagnostic figures to: {figure_dir}")


if __name__ == "__main__":
    main()