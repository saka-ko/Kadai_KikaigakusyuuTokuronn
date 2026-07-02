from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

def set_japanese_font():
    candidates = [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font in candidates:
        if font in available_fonts:
            rcParams["font.family"] = font
            rcParams["axes.unicode_minus"] = False
            print(f"Using Japanese font: {font}")
            return

    print("Japanese font was not found. Please install IPAexGothic or Noto Sans CJK JP.")
    rcParams["axes.unicode_minus"] = False

set_japanese_font()

out_dir = Path("results/figures/ppt_ready")
out_dir.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv("results/tables/cluster_summary.csv")
under10 = pd.read_csv("results/tables/under10_cluster_summary.csv")

order = ["1強型レース", "標準型レース", "混戦型レース"]
summary["cluster_name"] = pd.Categorical(summary["cluster_name"], categories=order, ordered=True)
summary = summary.sort_values("cluster_name")
under10["cluster_name"] = pd.Categorical(under10["cluster_name"], categories=order, ordered=True)
under10 = under10.sort_values("cluster_name")

def save_bar_with_labels(labels, values, title, ylabel, out_path, fmt, footer=None, ylim_pad=1.18):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title, fontsize=18, pad=14)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * ylim_pad)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt(v),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    if footer:
        fig.text(0.5, 0.01, footer, ha="center", fontsize=10)
        fig.subplots_adjust(bottom=0.18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

save_bar_with_labels(
    summary["cluster_name"].astype(str).tolist(),
    summary["favorite_odds_mean"].tolist(),
    "クラスタごとの1番人気平均オッズ",
    "平均単勝オッズ（倍）",
    out_dir / "favorite_odds_mean_by_cluster_jp.png",
    lambda v: f"{v:.2f}倍",
    footer="レース前情報としてモデルに入力した特徴量",
)

save_bar_with_labels(
    under10["cluster_name"].astype(str).tolist(),
    under10["under10_count_mean"].tolist(),
    "クラスタごとの単勝10倍以内の馬数",
    "平均頭数",
    out_dir / "under10_count_mean_by_cluster_jp.png",
    lambda v: f"{v:.2f}頭",
    footer="モデルには直接入力していない補足指標。勝負圏の馬の厚みを表す",
)

save_bar_with_labels(
    summary["cluster_name"].astype(str).tolist(),
    summary["trifecta_payout_median"].tolist(),
    "クラスタごとの三連単払戻中央値",
    "三連単払戻中央値（円）",
    out_dir / "trifecta_payout_median_by_cluster_jp.png",
    lambda v: f"{int(round(v)):,}円",
    footer="三連単払戻はモデル入力に使わず、クラスタ解釈のために後から比較",
)

base = summary["trifecta_payout_median"].iloc[0]
ratios = (summary["trifecta_payout_median"] / base).tolist()
save_bar_with_labels(
    summary["cluster_name"].astype(str).tolist(),
    ratios,
    "1強型を基準にした三連単払戻中央値の倍率",
    "1強型レース = 1.0",
    out_dir / "trifecta_payout_median_ratio_by_cluster_jp.png",
    lambda v: f"{v:.1f}倍",
    footer="標準型は約2.1倍、混戦型は約5.1倍の中央値",
    ylim_pad=1.25,
)
