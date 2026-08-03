from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
SOURCE = OUT
DATASETS = ("bank", "nyc", "bitcoin")
RATES = (5, 10, 20)
METHODS = ("CADE", "QE", "Certain", "Interval-Based", "Distribution-Centric")
AGGREGATION_METHODS = ("CADE", "QE", "Interval-Based", "Distribution-Centric")
NONAGGREGATION_METHODS = ("CADE", "QE", "Certain")
DATASET_LABELS = {
    "bank": "Bank Marketing",
    "nyc": "NYC Taxi Trips",
    "bitcoin": "Bitcoin Heist",
}


def plot_runtime(runtime: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11, "legend.fontsize": 8.5})
    positions = {"Aggregation": np.array([0.0, 1.0, 2.0]), "Non-aggregation": np.array([4.0, 5.0, 6.0])}
    colors = {"CADE": "#009E73", "QE": "#E69F00", "Certain": "#7B61A8", "Interval-Based": "#0057B8", "Distribution-Centric": "#B2185B"}
    hatches = {"CADE": "..", "QE": "//", "Certain": "\\\\", "Interval-Based": "", "Distribution-Centric": ""}
    width = 0.18
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=False)
    for axis, dataset in zip(axes, DATASETS):
        for query_type, centers in positions.items():
            methods = AGGREGATION_METHODS if query_type == "Aggregation" else NONAGGREGATION_METHODS
            offsets = {method: (index - (len(methods) - 1) / 2.0) * width for index, method in enumerate(methods)}
            for method in methods:
                values = []
                for rate in RATES:
                    row = runtime[runtime["dataset"].eq(dataset) & runtime["rate"].eq(rate) & runtime["query_type"].eq(query_type) & runtime["method"].eq(method)]
                    if len(row) != 1:
                        raise RuntimeError(f"Missing runtime for {dataset}, {rate}, {query_type}, {method}")
                    values.append(float(row.iloc[0]["mean_time_s"]))
                axis.bar(centers + offsets[method], values, width, color=colors[method], hatch=hatches[method], edgecolor="white", linewidth=0.5)
        axis.axvline(3.0, color="#777777", linestyle=(0, (2, 2)), linewidth=0.9)
        axis.set_yscale("log")
        axis.set_ylim(1e-3, 5e2)
        axis.set_yticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2])
        axis.set_title(DATASET_LABELS[dataset], fontweight="bold")
        axis.set_xticks(np.concatenate([positions["Aggregation"], positions["Non-aggregation"]]))
        axis.set_xticklabels(["5%", "10%", "20%", "5%", "10%", "20%"])
        axis.text(1.0, -0.30, "Aggregation", ha="center", transform=axis.get_xaxis_transform())
        axis.text(5.0, -0.30, "Non-aggregation", ha="center", transform=axis.get_xaxis_transform())
        axis.grid(axis="y", alpha=0.35, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean execution time (s)")
    figure.supxlabel("Missingness rate", y=0.005)
    handles = [Patch(facecolor=colors[method], hatch=hatches[method], edgecolor="white", label=method) for method in METHODS]
    figure.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02), columnspacing=1.0, handlelength=1.6)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.76, wspace=0.27)
    figure.savefig(OUT / "figure3_without_caex.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def formatted(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.4f}"


def write_table(quality: pd.DataFrame) -> None:
    metrics = ("Non-aggregation (TVD)", "Non-aggregation ($\\Delta w$)", "Aggregation (Accuracy \\%)", "Aggregation ($\\Delta w$)")
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Mean query quality on injected MCAR, MAR, and factorizable MNAR data. The baseline values are computed from completed MCAR and MAR aggregation queries.}", r"\label{tab:injected_query_quality}", r"\resizebox{\linewidth}{!}{%", r"\setlength{\tabcolsep}{1.5pt}", r"\begin{tabular}{llccccccccccccccc}", r"\toprule", r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Query type and metric} & \multicolumn{3}{c}{CADE} & \multicolumn{3}{c}{QE} & \multicolumn{3}{c}{Certain} & \multicolumn{3}{c}{Interval-Based} & \multicolumn{3}{c}{Distribution-Centric} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}\cmidrule(lr){15-17}", r" & & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% & 5\% & 10\% & 20\% \\", r"\midrule"]
    for dataset_index, dataset in enumerate(DATASETS):
        for metric_index, metric in enumerate(metrics):
            values = []
            for method in METHODS:
                for rate in RATES:
                    source_metric = "Aggregation (Coverage \\%)" if method == "Interval-Based" and metric == "Aggregation (Accuracy \\%)" else metric
                    row = quality[quality["dataset"].eq(dataset) & quality["rate"].eq(rate) & quality["method"].eq(method) & quality["metric"].eq(source_metric)]
                    if method == "Certain" and metric != "Non-aggregation (TVD)":
                        values.append("N/A")
                    elif method in ("Interval-Based", "Distribution-Centric") and metric.startswith("Non-aggregation"):
                        values.append("N/A")
                    elif method == "Distribution-Centric" and metric == "Aggregation ($\\Delta w$)":
                        values.append("N/A")
                    elif row.empty or int(row.iloc[0]["completed_query_count"]) == 0:
                        values.append("--")
                    elif method == "Interval-Based" and metric == "Aggregation (Accuracy \\%)":
                        values.append(r"100\%")
                    else:
                        values.append(formatted(float(row.iloc[0]["value"])))
            prefix = rf"\multirow{{4}}{{*}}{{{DATASET_LABELS[dataset]}}}" if metric_index == 0 else ""
            lines.append(prefix + " & " + metric + " & " + " & ".join(values) + r" \\")
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"))
    (OUT / "table3_quality_without_caex.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    runtime = pd.read_csv(SOURCE / "figure3_runtime.csv")
    quality = pd.read_csv(SOURCE / "table3_quality.csv")
    runtime = runtime[runtime["method"].isin(METHODS)].copy()
    quality = quality[quality["method"].isin(METHODS)].copy()
    runtime.to_csv(OUT / "figure3_runtime_without_caex.csv", index=False)
    quality.to_csv(OUT / "table3_quality_without_caex.csv", index=False)
    plot_runtime(runtime)
    write_table(quality)


if __name__ == "__main__":
    main()
