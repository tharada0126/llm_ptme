"""
correct_ptme.py

LLMベンチマーク（HumanEval / GSM8K）の実行プロファイルを集計・可視化するスクリプト。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# use science plot
import scienceplots  # type: ignore
plt.style.use("science")
# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

RESULT_DIR = Path("outputs1")

AXIS_LABELS: dict[str, str] = {
    "performance":        "$P$ (pass@1)",
    "time":               "$T$ (s)",
    "memory":             "$M$ (GB)",
    "energy":             "$E$ (kJ)",
    "Parameter Count":    "Params ($\\times 10^9$)",
    "Size":               "Size (GB)",
    "prompt_tokens":      "Input Tokens",
    "completion_tokens":  "Output Tokens",
    "total_tokens":       "Total Tokens",
}

PLOTLY_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4",
]

TOKEN_TYPES  = ["prompt_tokens", "completion_tokens", "total_tokens"]
TME_METRICS  = ["energy", "time", "memory"]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _safe_model_key(model_name: str) -> str:
    """モデル名をファイルパス用キーに変換する。"""
    return model_name.replace(":", "_").replace(".", "_")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def _mean_of(records: list[dict], key: str) -> float:
    values = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(values))


def _profile_path(benchmark: str, model_name: str) -> Path:
    key = _safe_model_key(model_name)
    prefix = "HE" if benchmark == "human_eval" else "GSM8K"
    return RESULT_DIR / f"{benchmark}_{key}" / f"{prefix}_{model_name}_profiles.jsonl"


def _results_path(benchmark: str, model_name: str) -> Path:
    key = _safe_model_key(model_name)
    if benchmark == "human_eval":
        return (
            RESULT_DIR
            / f"human_eval_{key}"
            / f"HE_{model_name}_samples_extracted.jsonl_results.jsonl"
        )
    return RESULT_DIR / f"gsm8k_{key}" / f"GSM8K_{model_name}_samples.jsonl"


def _correct_key(benchmark: str) -> str:
    return "passed" if benchmark == "human_eval" else "correct"


# ---------------------------------------------------------------------------
# プロファイル集計
# ---------------------------------------------------------------------------

def take_average(benchmark: str, model_name: str) -> dict[str, Any]:
    """1モデル分のプロファイルを平均し、正答率を付加して返す。"""
    profile_data = _read_jsonl(_profile_path(benchmark, model_name))
    results_data = _read_jsonl(_results_path(benchmark, model_name))

    correct_key   = _correct_key(benchmark)
    correct_count = sum(1 for r in results_data if r.get(correct_key, False))
    total_count   = len(results_data)

    print(f"Correct: {correct_count}/{total_count} ({correct_count / total_count:.2%})")

    profile_keys = [
        "cpu_energy", "dram_energy", "time",
        "prompt_tokens", "completion_tokens",
        "memory_peak", "rss_final_bytes",
    ]
    avg = {k: _mean_of(profile_data, k) for k in profile_keys}
    avg["model_name"]  = model_name
    avg["performance"] = correct_count / total_count

    print(f"Average profile for {model_name}: {avg}")
    return avg


# ---------------------------------------------------------------------------
# トークン vs T/M/E 相関
# ---------------------------------------------------------------------------

def _build_token_tme_df(result_dir: Path, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for model_name in model_names:
        key         = _safe_model_key(model_name)
        profile_dir = result_dir / f"human_eval_{key}"
        profiles    = _read_jsonl(profile_dir / f"HE_{model_name}_profiles.jsonl")
        results     = _read_jsonl(
            profile_dir / f"HE_{model_name}_samples_extracted.jsonl_results.jsonl"
        )
        for profile, _ in zip(profiles, results):
            pt = profile["prompt_tokens"]
            ct = profile["completion_tokens"]
            rows.append({
                "model_name":        model_name,
                "prompt_tokens":     pt,
                "completion_tokens": ct,
                "total_tokens":      pt + ct,
                "energy":            (profile["cpu_energy"] + profile["dram_energy"]) / 1_000,
                "time":              profile["time"],
                "memory":            (profile["memory_peak"] + profile["rss_final_bytes"]) / 1_024**3,
            })
    return pd.DataFrame(rows)


def _compute_correlations(df: pd.DataFrame, model_names: list[str]) -> dict:
    corrs: dict = {}
    for model_name in model_names:
        sub = df[df["model_name"] == model_name]
        corrs[model_name] = {
            tok: {met: sub[tok].corr(sub[met]) for met in TME_METRICS}
            for tok in TOKEN_TYPES
        }
    return corrs


def _save_corr_table(corrs: dict, model_names: list[str], output_dir: Path) -> None:
    multi_idx = pd.MultiIndex.from_product(
        [TOKEN_TYPES, TME_METRICS], names=["token_type", "metric"]
    )
    corr_df = pd.DataFrame(index=model_names, columns=multi_idx)
    for model in model_names:
        for tok in TOKEN_TYPES:
            for met in TME_METRICS:
                corr_df.loc[model, (tok, met)] = corrs[model][tok][met]

    corr_df.to_csv(output_dir / "tokens_tme_correlation.csv")

    corr_fmt = corr_df.copy()
    for col in corr_fmt.columns:
        corr_fmt[col] = corr_fmt[col].apply(lambda x: f"{x:.2f}")
    corr_fmt = corr_fmt.rename(columns=AXIS_LABELS)
    with (output_dir / "tokens_tme_correlation.tex").open("w") as f:
        f.write(corr_fmt.to_latex())


def _plot_token_vs_tme(df: pd.DataFrame, output_dir: Path) -> None:
    for tok in TOKEN_TYPES:
        for met in TME_METRICS:
            plt.figure()
            sns.scatterplot(
                data=df, x=tok, y=met, hue="model_name", style="model_name"
            )
            plt.xlabel(AXIS_LABELS[tok])
            plt.ylabel(AXIS_LABELS[met])
            plt.legend(title="Model Name")
            plt.tight_layout()
            plt.savefig(output_dir / f"{tok}_vs_{met}.pdf")
            plt.close()


def _plot_per_token_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    df = df.copy()
    df["time_per_token"]   = df["time"]   / df["total_tokens"]
    df["energy_per_token"] = df["energy"] / df["total_tokens"]

    for col, ylabel, fname in [
        ("time_per_token",   "Time per Token (s/token)",   "total_tokens_per_energy_time.pdf"),
        ("energy_per_token", "Energy per Token (kJ/token)", "total_tokens_per_energy_energy.pdf"),
    ]:
        plt.figure()
        sns.boxplot(data=df, x="model_name", y=col)
        plt.xlabel("Model Name")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / fname)
        plt.close()


def corr_between_tokens_and_time(
    result_dir: Path | str,
    model_names: list[str],
    output_dir: Path | str,
) -> None:
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)

    df    = _build_token_tme_df(result_dir, model_names)
    corrs = _compute_correlations(df, model_names)

    with (output_dir / "tokens_tme_correlation.json").open("w") as f:
        json.dump(corrs, f, indent=4)

    _save_corr_table(corrs, model_names, output_dir)
    _plot_token_vs_tme(df, output_dir)
    _plot_per_token_boxplots(df, output_dir)


# ---------------------------------------------------------------------------
# 3D インタラクティブプロット
# ---------------------------------------------------------------------------

def plot_3d_interactive(
    df: pd.DataFrame,
    labels: dict[str, str],
    output_dir: Path,
    benchmark: str,
) -> go.Figure:
    fig = go.Figure()

    for i, (model, group) in enumerate(df.groupby("model_name")):
        color = PLOTLY_COLORS[i % len(PLOTLY_COLORS)]

        fig.add_trace(go.Scatter3d(
            x=group["energy"],
            y=group["time"],
            z=group["performance"],
            mode="markers+text",
            name=model,
            marker=dict(size=5, symbol="circle", color=color, opacity=0.8),
            text=[model],
            textposition="top center",
        ))

        for _, row in group.iterrows():
            e, t, p = row["energy"], row["time"], row["performance"]
            # ドロップライン（各軸平面へ）
            for xs, ys, zs in [
                ([e, e], [t, t], [0, p]),
                ([e, e], [0, t], [p, p]),
                ([0, e], [t, t], [p, p]),
            ]:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=2),
                    showlegend=False, opacity=0.5,
                ))
            # 各平面への投影点
            for xe, ye, ze in [(e, t, 0), (e, 0, p), (0, t, p)]:
                fig.add_trace(go.Scatter3d(
                    x=[xe], y=[ye], z=[ze],
                    mode="markers",
                    marker=dict(size=3, color=color, opacity=0.4),
                    showlegend=False,
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title=labels["energy"],
            yaxis_title=labels["time"],
            zaxis_title=labels["performance"],
            zaxis=dict(range=[0, 100]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        title=f"3D Scatter: Performance vs Time vs Energy ({benchmark})",
        legend_title="Model",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    out_path = output_dir / "performance_time_energy_3d_interactive.html"
    pio.write_html(fig, str(out_path), include_plotlyjs="cdn")
    print(f"Interactive 3D plot saved to: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# 2D / 3D 散布図（matplotlib）
# ---------------------------------------------------------------------------

def _annotate_2d(ax, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    x_mean = df[x_col].mean()
    for _, row in df.iterrows():
        ha = "left" if row[x_col] < x_mean else "right"
        ax.text(row[x_col], row[y_col], row["model_name"], fontsize=10, ha=ha)


def _plot_performance_vs_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    for metric in TME_METRICS:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df, x="performance", y=metric,
            hue="model_name", style="model_name", s=100, ax=ax,
        )
        ax.set_xlabel(AXIS_LABELS["performance"])
        ax.set_ylabel(AXIS_LABELS[metric])
        _annotate_2d(ax, df, "performance", metric)
        ax.get_legend().remove()
        ax.set_xlim(0, 100)
        plt.tight_layout()
        plt.savefig(output_dir / f"performance_vs_{metric}.pdf")
        plt.close()


def _plot_3d_static(df: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")

    colors = sns.color_palette("hsv", df["model_name"].nunique())
    xlim1, ylim1 = None, None  # 確定後に使う

    for i, (model, group) in enumerate(df.groupby("model_name")):
        ax.scatter(
            group["energy"], group["time"], group["performance"],
            marker="o", s=100, label=model,
            color=colors[i % len(colors)], alpha=0.8,
        )

    # 軸範囲確定後にドロップライン描画
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_zlim(0, 100)

    for i, row in df.reset_index().iterrows():
        e, t, p = row["energy"], row["time"], row["performance"]
        kw = dict(color="gray", linestyle="dashed", alpha=0.5)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        zl = ax.get_zlim()
        ax.plot([e, e], [t, t], [0, p], **kw)
        ax.plot([e, e], [0, t], [p, p], **kw)
        ax.plot([0, e], [t, t], [p, p], **kw)
        color = colors[i % len(colors)]
        for xe, ye, ze in [(e, t, 0), (e, 0, p), (0, t, p)]:
            ax.scatter(xe, ye, ze, marker="o", s=10, color=color, alpha=0.5)

    # アノテーション
    t_mean = df["time"].mean()
    for _, row in df.reset_index().iterrows():
        ha = "left" if row["time"] < t_mean else "right"
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        zl = ax.get_zlim()
        span = lambda lim: (lim[1] - lim[0]) * 0.02
        sign = 1 if ha == "left" else -1
        ax.text(
            row["energy"] + sign * span(ax.get_xlim()),
            row["time"]   + sign * span(ax.get_ylim()),
            row["performance"] + sign * span(ax.get_zlim()),
            row["model_name"], fontsize=10, ha=ha,
        )

    ax.set_xlabel(AXIS_LABELS["energy"])
    ax.set_ylabel(AXIS_LABELS["time"])
    ax.set_zlabel(AXIS_LABELS["performance"])
    ax.grid(True, alpha=0.1)
    ax.view_init(azim=-45, elev=25)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_time_energy_3d.pdf")
    plt.close()


# ---------------------------------------------------------------------------
# モデル基本情報（ollama_meminfo.json）の読み込み
# ---------------------------------------------------------------------------

def load_model_basics(json_path: Path, model_names: list[str]) -> pd.DataFrame:
    """ollama_meminfo.json からパラメータ数とロード済みサイズを取得する。"""
    with json_path.open() as f:
        data = json.load(f)

    rows: dict[str, dict] = {}
    for info in data["models"]:
        name = info["name"]
        if name not in model_names:
            continue
        rows[name] = {
            "Parameter Count": info["details"]["parameter_count"] / 1e9,
            "Size":            info["loaded_memory"]["total_bytes"] / 1_024**3,
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "model_name"
    return df


# ---------------------------------------------------------------------------
# メイン集計・出力
# ---------------------------------------------------------------------------

def _build_average_df(all_averages: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_averages).set_index("model_name")
    df["energy"] = (df["cpu_energy"] + df["dram_energy"]) / 1_000       # J → kJ
    df["memory"] = (df["rss_final_bytes"] + df["memory_peak"]) / 1_024**3  # B → GB
    df.drop(columns=["cpu_energy", "dram_energy", "rss_final_bytes", "memory_peak"], inplace=True)
    df["performance"] *= 100  # 割合 → パーセント
    return df[["performance", "time", "memory", "energy", "prompt_tokens", "completion_tokens"]]


def _save_average_table(df: pd.DataFrame, output_dir: Path) -> None:
    df.to_csv(output_dir / "average_profiles.csv")

    df_fmt = df.copy()
    for col in df_fmt.columns:
        fmt = "{:.4f}" if col == "performance" else "{:.2f}"
        df_fmt[col] = df_fmt[col].apply(lambda x, f=fmt: f.format(x))

    with (output_dir / "average_profiles.tex").open("w") as f:
        f.write(df_fmt.to_latex())


def _plot_metric_vs_energy(merge_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ["Parameter Count", "Size", "prompt_tokens", "completion_tokens"]:
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=merge_df, x=metric, y="energy",
            hue="model_name", style="model_name", s=100, ax=ax,
        )
        ax.set_xlabel(AXIS_LABELS[metric])
        ax.set_ylabel(AXIS_LABELS["energy"])
        _annotate_2d(ax, merge_df, metric, "energy")
        ax.get_legend().remove()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_vs_energy.pdf")
        plt.close()


def run_benchmark(benchmark: str, model_names: list[str]) -> None:
    output_dir = RESULT_DIR / f"{benchmark}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 集計 ---
    all_averages = [take_average(benchmark, m) for m in model_names]
    with (output_dir / "average_profiles.json").open("w") as f:
        json.dump(all_averages, f, indent=4)

    df = _build_average_df(all_averages)
    _save_average_table(df, output_dir)

    # --- 散布図（2D） ---
    df_plot = pd.read_csv(output_dir / "average_profiles.csv")
    _plot_performance_vs_metrics(df_plot, output_dir)

    # --- 散布図（3D） ---
    plot_3d_interactive(df_plot, AXIS_LABELS, output_dir, benchmark)
    _plot_3d_static(df_plot, output_dir)

    # --- モデル基本情報とマージ ---
    basics_df = load_model_basics(Path("ollama_meminfo.json"), model_names)
    average_profiles = pd.read_csv(output_dir / "average_profiles.csv", index_col=0)
    combined = average_profiles.join(basics_df)[
        ["Parameter Count", "Size", "performance", "time", "memory", "energy",
         "prompt_tokens", "completion_tokens"]
    ].rename(columns=AXIS_LABELS)

    combined.to_csv(output_dir / "combined_results.csv")
    with (output_dir / "combined_results.tex").open("w") as f:
        f.write(combined.to_latex(index=True, float_format="%.2f", column_format="lrrrrrrrr"))

    merge_df = df_plot.merge(basics_df.reset_index(), on="model_name")
    _plot_metric_vs_energy(merge_df, output_dir)

    # --- トークン相関 ---
    corr_between_tokens_and_time(RESULT_DIR, model_names, output_dir)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_names = [
        "tinyllama:1.1b", "phi:2.7b", "phi3:3.8b",
        "qwen2.5:1.5b", "gemma:2b", "mistral:7b",
    ]
    for benchmark in ("human_eval", "gsm8k"):
        run_benchmark(benchmark, model_names)
