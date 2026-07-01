from pathlib import Path
import argparse

import numpy as np
import pandas as pd


RESULT_FILE = "19860105-20210731_race_result.csv"
ODDS_FILE = "19860105-20210731_odds.csv"


def read_race_result(raw_dir: Path, start_date: str) -> pd.DataFrame:
    """出走馬単位のレース結果データを読み込む。"""

    usecols = [
        "レースID",
        "レース日付",
        "競馬場名",
        "レース番号",
        "レース名",
        "芝・ダート区分",
        "距離(m)",
        "馬場状態1",
        "着順",
        "馬番",
        "単勝",
        "人気",
    ]

    path = raw_dir / RESULT_FILE
    df = pd.read_csv(path, encoding="utf-8", usecols=usecols)

    df["レース日付"] = pd.to_datetime(df["レース日付"], errors="coerce")

    for col in ["着順", "馬番", "単勝", "人気", "距離(m)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["レース日付"] >= pd.Timestamp(start_date)].copy()
    df = df.dropna(subset=["レースID", "単勝", "人気"])
    df["レースID"] = df["レースID"].astype("int64")

    return df


def build_race_features(result_df: pd.DataFrame) -> pd.DataFrame:
    """馬単位のデータから、レース単位の特徴量を作る。"""

    meta_cols = [
        "レース日付",
        "競馬場名",
        "レース番号",
        "レース名",
        "芝・ダート区分",
        "距離(m)",
        "馬場状態1",
    ]

    meta = result_df.groupby("レースID", sort=False)[meta_cols].first()
    field_size = result_df.groupby("レースID").size().rename("field_size")

    # 1〜3番人気の単勝オッズ
    fav = (
        result_df[result_df["人気"].isin([1, 2, 3])]
        .groupby(["レースID", "人気"])["単勝"]
        .min()
        .unstack("人気")
        .rename(
            columns={
                1.0: "favorite_odds",
                2.0: "second_favorite_odds",
                3.0: "third_favorite_odds",
            }
        )
    )

    top3_odds_mean = (
        result_df[result_df["人気"].isin([1, 2, 3])]
        .groupby("レースID")["単勝"]
        .mean()
        .rename("top3_odds_mean")
    )

    # オッズ分布のエントロピー
    # 単勝オッズの逆数を「市場が見ている勝率」に近い量として使う
    positive = result_df[result_df["単勝"] > 0].copy()
    positive["inv_odds"] = 1.0 / positive["単勝"].astype(float)
    positive["inv_log_inv"] = positive["inv_odds"] * np.log(positive["inv_odds"])

    sum_inv = positive.groupby("レースID")["inv_odds"].sum()
    sum_inv_log_inv = positive.groupby("レースID")["inv_log_inv"].sum()

    odds_entropy = (
        np.log(sum_inv) - (sum_inv_log_inv / sum_inv)
    ).rename("odds_entropy")

    # 勝ち馬
    winner = (
        result_df[result_df["着順"] == 1]
        .sort_values(["レースID", "馬番"])
        .groupby("レースID", sort=False)
        .agg(
            winner_popularity=("人気", "first"),
            winner_odds=("単勝", "first"),
            winner_horse_no=("馬番", "first"),
        )
    )

    features = (
        meta
        .join(field_size)
        .join(fav)
        .join(top3_odds_mean)
        .join(odds_entropy)
        .join(winner)
        .reset_index()
        .rename(
            columns={
                "レース日付": "race_date",
                "競馬場名": "racecourse",
                "レース番号": "race_no",
                "レース名": "race_name",
                "芝・ダート区分": "surface",
                "距離(m)": "distance",
                "馬場状態1": "track_condition",
            }
        )
    )

    features["odds_gap"] = (
        features["second_favorite_odds"] - features["favorite_odds"]
    )

    return features


def read_payouts(raw_dir: Path) -> pd.DataFrame:
    """払戻データを読み込む。odds.csv内の「オッズ」は払戻金として扱う。"""

    usecols = [
        "レースID",
        "単勝1_オッズ",
        "馬連1_オッズ",
        "馬単1_オッズ",
        "三連複1_オッズ",
        "三連単1_オッズ",
    ]

    path = raw_dir / ODDS_FILE
    df = pd.read_csv(path, encoding="utf-8", usecols=usecols)

    df = df.rename(
        columns={
            "単勝1_オッズ": "win_payout",
            "馬連1_オッズ": "quinella_payout",
            "馬単1_オッズ": "exacta_payout",
            "三連複1_オッズ": "trio_payout",
            "三連単1_オッズ": "trifecta_payout",
        }
    )

    df["レースID"] = df["レースID"].astype("int64")

    for col in [
        "win_payout",
        "quinella_payout",
        "exacta_payout",
        "trio_payout",
        "trifecta_payout",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def merge_features_and_payouts(
    race_features: pd.DataFrame,
    payouts: pd.DataFrame,
) -> pd.DataFrame:
    """レース特徴量と払戻データを結合する。"""

    df = race_features.merge(payouts, on="レースID", how="left")

    payout_cols = [
        "win_payout",
        "quinella_payout",
        "exacta_payout",
        "trio_payout",
        "trifecta_payout",
    ]

    for col in payout_cols:
        df[f"{col}_log"] = np.log1p(df[col])

    model_feature_cols = [
        "field_size",
        "favorite_odds",
        "second_favorite_odds",
        "odds_gap",
        "top3_odds_mean",
        "odds_entropy",
        "winner_popularity",
        "winner_odds",
        "win_payout_log",
        "quinella_payout_log",
        "trio_payout_log",
        "trifecta_payout_log",
    ]

    before = len(df)
    df = df.dropna(subset=model_feature_cols).copy()
    after = len(df)

    print(f"Removed rows with missing model features: {before - after}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/processed/race_features.csv")
    parser.add_argument("--start-date", type=str, default="2009-01-01")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading race result data...")
    result_df = read_race_result(raw_dir, args.start_date)

    print("Building race-level features...")
    race_features = build_race_features(result_df)

    print("Reading payout data...")
    payouts = read_payouts(raw_dir)

    print("Merging features and payouts...")
    df = merge_features_and_payouts(race_features, payouts)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_path}")
    print(f"Number of races: {len(df)}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()