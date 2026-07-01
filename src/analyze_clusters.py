from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "field_size",
    "distance",
    "favorite_odds",
    "second_favorite_odds",
    "third_favorite_odds",
    "odds_gap",
    "top3_odds_mean",
    "odds_entropy",
    "winner_popularity",
    "winner_odds",
    "win_payout",
    "quinella_payout",
    "trio_payout",
    "trifecta_payout",
    "win_payout_log",
    "quinella_payout_log",
    "trio_payout_log",
    "trifecta_payout_log",
]


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """クラスタごとの平均・中央値・件数を集計する。"""

    summary = df.groupby("cluster")[SUMMARY_COLUMNS].agg(["mean", "median"])
    counts = df.groupby("cluster").size().rename(("race_count", ""))

    summary = pd.concat([counts, summary], axis=1)

    # 見やすいようにMultiIndex列を平坦化する
    summary.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in summary.columns
    ]

    return summary.reset_index()


def assign_cluster_names(summary: pd.DataFrame) -> dict[int, str]:
    """
    レース前のオッズ構造に基づいてクラスタ名を付ける。
    odds_entropyが小さいほど1強型、大きいほど混戦型と解釈する。
    """

    sorted_clusters = summary.sort_values("odds_entropy_mean")["cluster"].tolist()

    names = {}
    if len(sorted_clusters) >= 1:
        names[sorted_clusters[0]] = "1強型レース"
    if len(sorted_clusters) >= 2:
        names[sorted_clusters[1]] = "標準型レース"
    if len(sorted_clusters) >= 3:
        names[sorted_clusters[2]] = "混戦型レース"

    for c in sorted_clusters[3:]:
        names[c] = "その他"

    return names


def save_cluster_summary_plot(summary: pd.DataFrame, out_path: Path) -> None:
    """クラスタごとの主要指標を棒グラフで保存する。"""

    plot_df = summary.copy()
    plot_df["cluster_label"] = plot_df["cluster"].astype(str)

    metrics = [
        ("winner_popularity_mean", "Average winner popularity"),
        ("favorite_odds_mean", "Average favorite odds"),
        ("trifecta_payout_log_mean", "Average log trifecta payout"),
    ]

    for col, title in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(plot_df["cluster_label"], plot_df[col])
        plt.xlabel("Cluster")
        plt.ylabel(col)
        plt.title(title)
        plt.tight_layout()

        metric_out = out_path.parent / f"{out_path.stem}_{col}.png"
        plt.savefig(metric_out, dpi=200)
        plt.close()


def save_scatter_plot(df: pd.DataFrame, out_path: Path) -> None:
    """1番人気オッズと三連単払戻logの散布図をクラスタ色分けで保存する。"""

    plt.figure(figsize=(8, 6))

    for cluster_id, group in df.groupby("cluster"):
        plt.scatter(
            group["favorite_odds"],
            group["trifecta_payout_log"],
            s=8,
            alpha=0.5,
            label=f"Cluster {cluster_id}",
        )

    plt.xlabel("Favorite odds")
    plt.ylabel("Log trifecta payout")
    plt.title("Bayesian GMM clustering result")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_winner_popularity_plot(df: pd.DataFrame, out_path: Path) -> None:
    """勝ち馬人気の分布をクラスタごとに保存する。"""

    plt.figure(figsize=(8, 6))

    clusters = sorted(df["cluster"].unique())
    data = [df[df["cluster"] == c]["winner_popularity"].dropna() for c in clusters]

    plt.boxplot(data, tick_labels=[f"Cluster {c}" for c in clusters], showfliers=False)
    plt.xlabel("Cluster")
    plt.ylabel("Winner popularity")
    plt.title("Winner popularity by cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/tables/race_features_with_cluster.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/tables/cluster_summary.csv",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="results/figures",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    summary_output_path = Path(args.summary_output)
    figure_dir = Path(args.figure_dir)

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clustered race data...")
    df = pd.read_csv(input_path)

    print("Summarizing clusters...")
    summary = summarize_clusters(df)

    cluster_names = assign_cluster_names(summary)
    summary["cluster_name"] = summary["cluster"].map(cluster_names)

    df["cluster_name"] = df["cluster"].map(cluster_names)

    summary.to_csv(summary_output_path, index=False, encoding="utf-8-sig")
    df.to_csv(input_path, index=False, encoding="utf-8-sig")

    print("Cluster summary:")

    display_cols = [
        "cluster",
        "cluster_name",
        "race_count",
        "field_size_mean",
        "distance_mean",
        "favorite_odds_mean",
        "second_favorite_odds_mean",
        "third_favorite_odds_mean",
        "odds_gap_mean",
        "top3_odds_mean_mean",
        "odds_entropy_mean",
        "winner_popularity_mean",
        "winner_odds_mean",
        "trifecta_payout_median",
        "trifecta_payout_log_mean",
    ]

    display_cols = [col for col in display_cols if col in summary.columns]

    print(summary[display_cols].to_string(index=False))

    print("Saving figures...")
    save_scatter_plot(
        df,
        figure_dir / "clustering_result.png",
    )
    save_winner_popularity_plot(
        df,
        figure_dir / "winner_popularity_by_cluster.png",
    )
    save_cluster_summary_plot(
        summary,
        figure_dir / "cluster_summary.png",
    )

    print(f"Saved summary: {summary_output_path}")
    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()