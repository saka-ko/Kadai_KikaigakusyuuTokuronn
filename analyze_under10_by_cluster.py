from pathlib import Path
import pandas as pd


RAW_RACE_RESULT = "data/raw/19860105-20210731_race_result.csv"
CLUSTERED_RACE = "results/tables/race_features_with_cluster.csv"
OUT_LABELED = "results/tables/race_features_with_cluster_under10.csv"
OUT_SUMMARY = "results/tables/under10_cluster_summary.csv"


def main():
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clustered race data...")
    race_df = pd.read_csv(CLUSTERED_RACE)

    print("Loading raw race result data...")
    raw_df = pd.read_csv(
        RAW_RACE_RESULT,
        usecols=["レースID", "単勝"],
        low_memory=False,
    )

    raw_df["レースID"] = pd.to_numeric(raw_df["レースID"], errors="coerce")
    raw_df["単勝"] = pd.to_numeric(raw_df["単勝"], errors="coerce")

    raw_df = raw_df.dropna(subset=["レースID", "単勝"]).copy()
    raw_df["レースID"] = raw_df["レースID"].astype("int64")
    raw_df = raw_df[raw_df["単勝"] > 0].copy()

    print("Calculating under-10-odds features...")

    total_stats = raw_df.groupby("レースID").agg(
        total_horses=("単勝", "size"),
        all_odds_mean=("単勝", "mean"),
    ).reset_index()

    under10 = raw_df[raw_df["単勝"] <= 10.0].copy()

    under10_stats = under10.groupby("レースID").agg(
        under10_count=("単勝", "size"),
        under10_mean_odds=("単勝", "mean"),
        under10_max_odds=("単勝", "max"),
        under10_min_odds=("単勝", "min"),
    ).reset_index()

    odds_stats = total_stats.merge(under10_stats, on="レースID", how="left")

    fill_cols = [
        "under10_count",
        "under10_mean_odds",
        "under10_max_odds",
        "under10_min_odds",
    ]

    for col in fill_cols:
        odds_stats[col] = odds_stats[col].fillna(0)

    odds_stats["under10_ratio"] = (
        odds_stats["under10_count"] / odds_stats["total_horses"]
    )

    race_df["レースID"] = pd.to_numeric(race_df["レースID"], errors="coerce").astype("int64")

    merged = race_df.merge(odds_stats, on="レースID", how="left")

    merged.to_csv(OUT_LABELED, index=False)
    print(f"Saved labeled data: {OUT_LABELED}")

    print("Making summary...")

    summary = merged.groupby(["cluster", "cluster_name"]).agg(
        race_count=("レースID", "size"),
        under10_count_mean=("under10_count", "mean"),
        under10_count_median=("under10_count", "median"),
        under10_ratio_mean=("under10_ratio", "mean"),
        favorite_odds_mean=("favorite_odds", "mean"),
        odds_gap_mean=("odds_gap", "mean"),
        odds_entropy_mean=("odds_entropy", "mean"),
        trifecta_payout_median=("trifecta_payout", "median"),
    ).reset_index()

    summary = summary.sort_values("cluster").reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"Saved summary: {OUT_SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()